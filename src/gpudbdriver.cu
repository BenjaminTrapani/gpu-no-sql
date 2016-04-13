#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"
#include <time.h>
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/sequence.h"
#include "thrust/transform.h"
#include "thrust/device_ptr.h"

using namespace GPUDB;

GPUDBDriver::GPUDBDriver(){
    int nDevices;
    const unsigned int gb = 1024 * 1024 * 1024;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDAGPUNoSQLDB starting...\n");
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  GPU Memory (GB): %f\n", ((float)prop.totalGlobalMem)/((float)(gb)));
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    }
    cudaDeviceProp propOfInterest;
    cudaGetDeviceProperties(&propOfInterest, 0);
    size_t memBytes = propOfInterest.totalGlobalMem;
    size_t allocSize = memBytes*0.05f;
    numEntries = allocSize/sizeof(CoreTupleType);
    printf("Num entries = %i\n", numEntries);

    //buffer allocation and initialization
    deviceEntries.reserve(numEntries);
    deviceIntermediateBuffer1 = new DeviceVector_t(numEntries);
    deviceIntermediateBuffer2 = new DeviceVector_t(numEntries);
    hostResultBuffer = new HostVector_t(numEntries);
}
GPUDBDriver::~GPUDBDriver(){
    delete deviceIntermediateBuffer1;
    deviceIntermediateBuffer1=0;

    delete deviceIntermediateBuffer2;
    deviceIntermediateBuffer2=0;

    delete hostResultBuffer;
    hostResultBuffer=0;
}

void GPUDBDriver::create(const CoreTupleType &object){
    deviceEntries.push_back(object);
}

void GPUDBDriver::create(const Doc & toCreate){
    create(toCreate.kvPair);
    for (std::vector<Doc>::const_iterator iter = toCreate.children.begin(); iter != toCreate.children.end(); ++iter) {
        create(*iter);
    }
}

void GPUDBDriver::batchCreate(const std::vector<Doc> & docs){
    for(std::vector<Doc>::const_iterator iter = docs.begin(); iter != docs.end(); ++iter){
        create(*iter);
    }
}

struct IsPartialTupleMatch : thrust::unary_function<CoreTupleType,bool>{
    inline IsPartialTupleMatch(const CoreTupleType & filter):_filter(filter){}

    __device__ __host__
    inline bool operator()(const CoreTupleType & val)const{
        return val == _filter;
    }

private:
    const CoreTupleType _filter;
};

struct IsFullTupleMatch : thrust::unary_function<CoreTupleType, bool>{
    inline IsFullTupleMatch(const CoreTupleType & filter):_filter(filter){}

    __device__ __host__
    inline bool operator()(const CoreTupleType & val)const{
        return val.fullCompare(_filter);
    }

private:
    CoreTupleType _filter;
};

struct ExtractParentID : thrust::unary_function<CoreTupleType, GPUSizeType>{
    __device__ __host__
    inline GPUSizeType operator() (const CoreTupleType & val)const{
        return val.parentID;
    }
};

struct ModifyTuple : thrust::unary_function<CoreTupleType, CoreTupleType>{
    inline ModifyTuple(const CoreTupleType & updates):_updates(updates){}

    __device__ __host__
    inline CoreTupleType operator() (const CoreTupleType & val)const{
        CoreTupleType result = val;
        result.data.bigVal = _updates.data.bigVal;
        result.valType = _updates.valType;
        return result;
    }
private:
    const CoreTupleType _updates;
};

struct FetchTupleWithParentIDs : thrust::unary_function<CoreTupleType,bool>{
    inline FetchTupleWithParentIDs(CoreTupleType* validIndices,
                                   const size_t indexToExamine, const CoreTupleType & filter):
            _validIndices(validIndices), _indexToExamine(indexToExamine), _filter(filter){
    }

    __device__ __host__
    inline bool operator()(const CoreTupleType & ival)const{
        return _validIndices[_indexToExamine].parentID == ival.id && ival == _filter;
    }

private:
    CoreTupleType * _validIndices;
    const size_t _indexToExamine;
    const CoreTupleType _filter;
};

struct FetchDescendentTuple : thrust::unary_function<CoreTupleType, bool>{
    inline FetchDescendentTuple(const CoreTupleType * desiredParentID): _desiredParentID(desiredParentID){}

    __device__ __host__
    inline bool operator()(const CoreTupleType & ival)const{
        return ival.parentID == _desiredParentID->id && ival.parentID!=0;
    }

private:
    const CoreTupleType * _desiredParentID;
};


void GPUDBDriver::update(const CoreTupleType &searchFilter, const CoreTupleType &updates){
    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), ModifyTuple(updates),
                         IsFullTupleMatch(searchFilter));
}
void GPUDBDriver::deleteBy(const CoreTupleType &searchFilter){
    thrust::remove_if(deviceEntries.begin(), deviceEntries.end(), IsFullTupleMatch(searchFilter));
}

void GPUDBDriver::searchEntries(const CoreTupleType & filter, DeviceVector_t * resultsFromThisStage,
                   DeviceVector_t * resultsFromLastStage,
                   const size_t numToSearch,
                   size_t &numFound){

    DeviceVector_t::iterator lastIter;
    numFound = 0;
    for(size_t i = 0; i < numToSearch; i++){
        lastIter = copy_if(deviceEntries.begin(), deviceEntries.begin() + deviceEntries.size(),
                                resultsFromThisStage->begin() + numFound, FetchTupleWithParentIDs(
                                thrust::raw_pointer_cast(resultsFromLastStage->data()),
                                i,
                                filter));
        if(lastIter != resultsFromThisStage->end())
            numFound += thrust::distance(resultsFromThisStage->begin()+numFound, lastIter);
    }
}

QueryResult GPUDBDriver::getRootsForFilterSet(const std::vector<CoreTupleType> & filters){
    DeviceVector_t::iterator lastIter = copy_if(deviceEntries.begin(), deviceEntries.begin() + deviceEntries.size(),
                                                deviceIntermediateBuffer1->begin(),
                                                IsPartialTupleMatch(filters[0]));
    size_t lastNumFound = thrust::distance(deviceIntermediateBuffer1->begin(), lastIter);
    size_t curNumFound = 0;

    DeviceVector_t * mostRecentResult = deviceIntermediateBuffer1;

    for(std::vector<CoreTupleType>::const_iterator iter = filters.begin()+1; iter != filters.end(); ++iter){
        size_t iterDistance = std::distance(filters.begin(), iter);
        if(iterDistance % 2 == 0){
            searchEntries(*iter, deviceIntermediateBuffer1, deviceIntermediateBuffer2, lastNumFound, curNumFound);
            mostRecentResult = deviceIntermediateBuffer1;
        }else{
            searchEntries(*iter, deviceIntermediateBuffer2, deviceIntermediateBuffer1, lastNumFound, curNumFound);
            mostRecentResult = deviceIntermediateBuffer2;
        }
        if(curNumFound==0) {
            break;
        }else{
            lastNumFound = curNumFound;
        }
    }

    QueryResult result;
    if(lastNumFound!=0) {
        result.numItems = lastNumFound;
        result.deviceResultPointer = mostRecentResult;
    }else{
        result.numItems = 0;
        result.deviceResultPointer = 0;
    }
    return result;
}

void GPUDBDriver::getEntriesForRoots(const QueryResult & rootResult, std::vector<Doc> & result){

    DeviceVector_t::iterator lastIter;
    size_t numFound = 0;
    thrust::copy(rootResult.deviceResultPointer->begin() + rootResult.beginOffset,
                    rootResult.deviceResultPointer->begin() + rootResult.beginOffset + rootResult.numItems,
                 hostResultBuffer->begin() + rootResult.beginOffset);

    size_t iterIndex = 0;
    for(DeviceVector_t::const_iterator iter = rootResult.deviceResultPointer->begin() + rootResult.beginOffset;
        iter != rootResult.deviceResultPointer->begin() + rootResult.beginOffset + rootResult.numItems; ++iter){

        Doc curDocVal((*hostResultBuffer)[rootResult.beginOffset + iterIndex]);
        result.push_back(curDocVal);

        DeviceVector_t::iterator destIter = rootResult.deviceResultPointer->begin() + rootResult.beginOffset + numFound + rootResult.numItems;
        lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.begin() + deviceEntries.size(),
                                   destIter,
                                   FetchDescendentTuple(thrust::raw_pointer_cast(&(*iter))));
        size_t mostRecentFoundCount = thrust::distance(destIter, lastIter);
        numFound += mostRecentFoundCount;

        if(mostRecentFoundCount) {
            QueryResult newQuery;
            newQuery.deviceResultPointer = rootResult.deviceResultPointer;
            newQuery.beginOffset = rootResult.beginOffset + numFound;
            newQuery.numItems = mostRecentFoundCount;
            getEntriesForRoots(newQuery, result[result.size()-1].children);
        }

        iterIndex++;
    }
}

std::vector<Doc> GPUDBDriver::getEntriesForRoots(const QueryResult & rootResult){
    std::vector<Doc> result;
    getEntriesForRoots(rootResult, result);

    return result;
}

std::vector<Doc> GPUDBDriver::getDocumentsForFilterSet(const std::vector<CoreTupleType> & filters){
    QueryResult rootResult = getRootsForFilterSet(filters);

    if(rootResult.numItems)
        return getEntriesForRoots(rootResult);

    return std::vector<Doc>(0);
}

int main(int argc, char * argv[]){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));
    Doc coreDoc;
    for(unsigned int i = 0; i < driver.getTableSize()-2; i++){
        Entry anEntry;
        anEntry.data.bigVal=0;
        anEntry.valType = GPUDB_BGV;
        anEntry.key=i;
        anEntry.id = i;
        coreDoc.children.push_back(Doc(anEntry));
    }
    Entry lastEntry;
    lastEntry.valType = GPUDB_BGV;
    lastEntry.data.bigVal = 1;
    lastEntry.key = 10;
    lastEntry.parentID = 3;
    coreDoc.children[3].children.push_back(lastEntry);

    Entry realLastEntry;
    realLastEntry.valType = GPUDB_BGV;
    realLastEntry.id = 51;
    realLastEntry.data.bigVal = 1;
    realLastEntry.key = 10;
    realLastEntry.parentID = 6;
    coreDoc.children[6].children.push_back(realLastEntry);

    driver.create(coreDoc);

    Entry filter1 = realLastEntry;
    Entry filter2;
    filter2.data.bigVal=0;
    filter2.valType = GPUDB_BGV;
    filter2.key=realLastEntry.parentID;

    std::vector<Entry> filters;
    filters.push_back(filter1);
    filters.push_back(filter2);

    clock_t t1, t2;

    t1 = clock();
    std::vector<Doc> hostqueryResult = driver.getDocumentsForFilterSet(filters);
    t2 = clock();

    float diff1 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device multi-filter query latency = %fms\n", diff1);

    for(std::vector<Doc>::iterator iter = hostqueryResult.begin(); iter != hostqueryResult.end(); ++iter){
        printf("Doc id = %llu\n", iter->kvPair.id);
        for(std::vector<Doc>::iterator nestedIter = iter->children.begin(); nestedIter != iter->children.end();
            ++nestedIter){
            printf("  child id = %llu\n", nestedIter->kvPair.id);
            Entry newEntry = nestedIter->kvPair;
            newEntry.data.bigVal = 52;
            t1 = clock();
            driver.update(nestedIter->kvPair, newEntry);
            t2 = clock();
            float diff2 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
            printf("update single element latency = %fms\n", diff2);

            std::vector<CoreTupleType> filterSet;
            filterSet.push_back(newEntry);
            std::vector<Doc> updatedElement = driver.getDocumentsForFilterSet(filterSet);
            for(std::vector<Doc>::iterator updatedIter = updatedElement.begin();
                    updatedIter != updatedElement.end(); ++updatedIter){
                printf("Updated value for id %llu = %lld\n", updatedIter->kvPair.id, updatedIter->kvPair.data.bigVal);
            }
        }
    }
    t1 = clock();
    driver.deleteBy(lastEntry);
    t2 = clock();
    float deleteDiff = ((float)(t2 - t1) / 1000000.0F ) * 1000;

    std::vector<CoreTupleType> searchForLastEntry;
    searchForLastEntry.push_back(lastEntry);
    std::vector<Doc> lastEntryResult = driver.getDocumentsForFilterSet(searchForLastEntry);
    if(lastEntryResult.size() == 0){
        printf("Successfully deleted last entry. Delete took %fms.\n", deleteDiff);
    }else{
        printf("Delete of last entry failed, still present in table.\n");
        for(std::vector<Doc>::iterator iter = lastEntryResult.begin(); iter != lastEntryResult.end(); ++iter){
            printf("Remaining id = %llu\n", iter->kvPair.id);
        }
    }

    return 0;
}