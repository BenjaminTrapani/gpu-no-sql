#include "gpudbdriver.h"
#include "Functors.h"

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
    size_t allocSize = memBytes*0.12f;
    numEntries = allocSize/sizeof(CoreTupleType);
    printf("Num entries = %i\n", numEntries);

    //buffer allocation and initialization
    deviceEntries.reserve(numEntries);

    deviceIntermediateBuffer1 = new DeviceVector_t(numEntries);
    //deviceIntermediateBuffer2 = new DeviceVector_t(numEntries);
    hostResultBuffer = new HostVector_t(numEntries);
    hostCreateBuffer = new HostVector_t();
    hostCreateBuffer->reserve(numEntries);
}
GPUDBDriver::~GPUDBDriver(){
    delete deviceIntermediateBuffer1;
    deviceIntermediateBuffer1=0;

    //delete deviceIntermediateBuffer2;
    //deviceIntermediateBuffer2=0;

    delete hostResultBuffer;
    hostResultBuffer=0;

    delete hostCreateBuffer;
    hostCreateBuffer = 0;
}

void GPUDBDriver::create(const CoreTupleType &object){
    hostCreateBuffer->push_back(object);
    //deviceEntries.push_back(object);
}

void GPUDBDriver::syncCreates(){
    DeviceVector_t::iterator oldEnd = deviceEntries.end();
    deviceEntries.resize(deviceEntries.size() + hostCreateBuffer->size());
    thrust::copy(hostCreateBuffer->begin(), hostCreateBuffer->end(), oldEnd);
    hostCreateBuffer->clear();
}

void GPUDBDriver::create(const Doc & toCreate) {
    create(toCreate.kvPair);
    for (std::vector<Doc>::const_iterator iter = toCreate.children.begin(); iter != toCreate.children.end(); ++iter) {
        create(*iter);
    }
}

void GPUDBDriver::batchCreate(std::vector<Doc> & docs){
    for(std::vector<Doc>::iterator iter = docs.begin(); iter != docs.end(); ++iter){
        create(*iter);
    }
}

void GPUDBDriver::update(const CoreTupleType &searchFilter, const CoreTupleType &updates){
    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), ModifyTuple(updates),
                         IsFullTupleMatch(searchFilter));
}
void GPUDBDriver::deleteBy(const CoreTupleType &searchFilter){
    thrust::remove_if(deviceEntries.begin(), deviceEntries.end(), IsFullTupleMatch(searchFilter));
}


void GPUDBDriver::optimizedSearchEntries(const FilterGroup & filterGroup, const unsigned long int layer){
    for(FilterGroup::const_iterator filterIter = filterGroup.begin(); filterIter != filterGroup.end();
        ++filterIter){
        DeviceVector_t::iterator lastIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                            IsTupleSelected(layer));
        while(lastIter != deviceEntries.end()){
            thrust::transform_if(deviceEntries.begin(), deviceEntries.end(),
                                 deviceEntries.begin(),
                                 SelectTuple(layer+1),
                                 FetchTupleWithParentID(*filterIter,
                                                    thrust::raw_pointer_cast(&(*lastIter))));

            lastIter = thrust::find_if(lastIter+1, deviceEntries.end(),
                                       IsTupleSelected(layer));
        }
    }
}

QueryResult GPUDBDriver::optimizedGetRootsForFilterSet(const FilterSet & filters){
    thrust::transform(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), UnselectTuple());
    for(FilterGroup::const_iterator firstGroupIter = filters[0].begin(); firstGroupIter != filters[0].end();
        ++firstGroupIter){
        thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), SelectTuple(1),
                             IsPartialTupleMatch(*firstGroupIter));
    }
    unsigned long int layer = 1;
    for(FilterSet::const_iterator iter = filters.begin()+1; iter != filters.end(); ++iter){
        optimizedSearchEntries(*iter, layer);
        layer++;
    }

    DeviceVector_t::iterator lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.end(),
                                                    deviceIntermediateBuffer1->begin(), IsTupleSelected(layer));


    QueryResult result;
    result.numItems = thrust::distance(deviceIntermediateBuffer1->begin(), lastIter);
    result.deviceResultPointer = deviceIntermediateBuffer1;
    result.beginOffset = 0;
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

std::vector<Doc> GPUDBDriver::getDocumentsForFilterSet(const FilterSet & filters){
    clock_t t1, t2;

    t1 = clock();
    QueryResult rootResult = optimizedGetRootsForFilterSet(filters);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    //printf("get roots for filter set took %fms\n", diff);

    if(rootResult.numItems)
        return getEntriesForRoots(rootResult);

    return std::vector<Doc>(0);
}
void generateNestedDoc(size_t nestings, Doc * parent, size_t beginIndex){
    Entry curVal;
    curVal.key = beginIndex;
    curVal.valType = GPUDB_BGV;
    curVal.data.bigVal = beginIndex;
    curVal.id = beginIndex;

    Doc intermediate(curVal);
    Doc * permIntermediate = parent->addChild(intermediate);

    if(nestings>0)
        generateNestedDoc(nestings-1, permIntermediate, beginIndex+1);
}
void runDeepNestingTests(){
    printf("Beginning deep nesting test:\n");

    GPUDBDriver driver;

    for(size_t i = 2; i < driver.getTableSize(); i+=5){
        Doc root;
        root.kvPair.key=i;
        root.kvPair.data.bigVal=i;
        root.kvPair.valType=GPUDB_BGV;
        root.kvPair.id = i;
        root.kvPair.parentID = 0;
        generateNestedDoc(3, &root, i+1);
        driver.create(root);
    }
    driver.syncCreates();
    printf("Database has %i entries\n", driver.getNumEntries());

    FilterSet filterByFirstFourNest;
    filterByFirstFourNest.reserve(4);
    for(int i = 5; i >= 2; i--){
        Entry curFilter;
        curFilter.key = i;
        curFilter.valType=GPUDB_BGV;
        curFilter.data.bigVal = i;
        FilterGroup curGroup;
        curGroup.push_back(curFilter);
        filterByFirstFourNest.push_back(curGroup);
    }

    clock_t t1, t2;

    t1 = clock();
    std::vector<Doc> result = driver.getDocumentsForFilterSet(filterByFirstFourNest);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Deep filter took %fms\n", diff);
    printf("Num results = %i\n", result.size());
    for(std::vector<Doc>::iterator iter = result.begin(); iter != result.end(); ++iter){
        printf(iter->toString().c_str());
    }

    printf("Deep nesting test finished.\n\n");
}

int main(int argc, char * argv[]){
    runDeepNestingTests();

    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));
    Doc coreDoc;
    for(unsigned int i = 0; i < driver.getTableSize()-3; i++){
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
    driver.syncCreates();
    printf("Database has %i entries\n", driver.getNumEntries());

    Entry filter1 = realLastEntry;
    Entry filter2;
    filter2.data.bigVal=0;
    filter2.valType = GPUDB_BGV;
    filter2.key=realLastEntry.parentID;

    FilterGroup filters1;
    FilterGroup filters2;
    filters1.push_back(filter1);
    filters2.push_back(filter2);

    FilterSet filterSet;
    filterSet.push_back(filters1);
    filterSet.push_back(filters2);

    clock_t t1, t2;
    t1 = clock();
    std::vector<Doc> hostqueryResult = driver.getDocumentsForFilterSet(filterSet);
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

            FilterGroup filterGroup;
            filterGroup.push_back(newEntry);
            FilterSet toCheck;
            toCheck.push_back(filterGroup);
            std::vector<Doc> updatedElement = driver.getDocumentsForFilterSet(toCheck);
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

    FilterGroup searchForLastEntry;
    searchForLastEntry.push_back(lastEntry);
    FilterSet searchForLastEntryFilter;
    searchForLastEntryFilter.push_back(searchForLastEntry);
    std::vector<Doc> lastEntryResult = driver.getDocumentsForFilterSet(searchForLastEntryFilter);
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