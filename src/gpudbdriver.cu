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
typedef GPUDBDriver::CoreTupleType CoreTupleType;
typedef GPUDBDriver::GPUSizeType GPUSizeType;

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
    deviceIntermediateBuffer=0;

    delete deviceIntermediateBuffer2;
    deviceIntermediateBuffer2=0;

    delete hostResultBuffer;
    hostResultBuffer=0;
}

void GPUDBDriver::create(const CoreTupleType &object){
    deviceEntries.push_back(object);
}

struct IsPartialTupleMatch : thrust::unary_function<GPUDBDriver::CoreTupleType,bool>{
    typedef GPUDBDriver::CoreTupleType CoreTupleType;


    inline IsPartialTupleMatch(const CoreTupleType & filter):_filter(filter){}

    __device__ __host__
    inline bool operator()(const CoreTupleType & val)const{
        return val == _filter;
    }

private:
    const CoreTupleType _filter;
};

struct ExtractParentID : thrust::unary_function<GPUDBDriver::CoreTupleType, GPUDBDriver::GPUSizeType>{
    __device__ __host__
    inline GPUDBDriver::GPUSizeType operator() (const CoreTupleType & val)const{
        return val.parentID;
    }
};

struct FetchTupleWithParentIDs : thrust::unary_function<GPUDBDriver::CoreTupleType,bool>{
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

void GPUDBDriver::update(const CoreTupleType &searchFilter, const CoreTupleType &updates){

}
void GPUDBDriver::deleteBy(const CoreTupleType &searchFilter){

}

void GPUDBDriver::sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter){
}

void GPUDBDriver::searchEntries(const CoreTupleType & filter, DeviceVector_t * resultsFromThisStage,
                   DeviceVector_t * resultsFromLastStage,
                   const size_t numToSearch,
                   size_t &numFound){

    DeviceVector_t::iterator lastIter;
    numFound = 0;
    for(size_t i = 0; i < numToSearch; i++){
        lastIter = copy_if(deviceEntries.begin(), deviceEntries.begin() + numEntries,
                                resultsFromThisStage->begin() + numFound, FetchTupleWithParentIDs(
                                thrust::raw_pointer_cast(resultsFromLastStage->data()),
                                i,
                                filter));
        if(lastIter != resultsFromThisStage->end())
            numFound += thrust::distance(resultsFromThisStage->begin()+numFound, lastIter);
    }
}

thrust::host_vector<CoreTupleType> * GPUDBDriver::getEntriesForFilterSet(std::vector<CoreTupleType> filters){
    DeviceVector_t::iterator lastIter = copy_if(deviceEntries.begin(), deviceEntries.begin() + numEntries,
                                                deviceIntermediateBuffer1->begin(),
                                                IsPartialTupleMatch(filter));
    size_t lastNumFound = thrust::distance(deviceIntermediateBuffer1->begin(), lastIter);
    size_t curNumFound = 0;

    DeviceVector_t * mostRecentResult = 0;

    for(std::vector<CoreTupleType>::iterator iter = filters.begin(); iter != filters.end(); ++iter){
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

    if(lastNumFound!=0) {
        printf("lastNumFound=%i\n", lastNumFound);
        *hostResultBuffer = *mostRecentResult;
        hostResultBuffer->resize(lastNumFound);
        return hostResultBuffer;
    }else{
        return 0;
    }
}

int main(int argc, char * argv[]){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));

    for(unsigned int i = 0; i < driver.getTableSize()-2; i++){
        Entry anEntry;
        anEntry.data.bigVal=0;
        anEntry.valType = GPUDB_BGV;
        anEntry.key=i;
        anEntry.id = i;
        driver.create(anEntry);
    }
    Entry lastEntry;
    lastEntry.valType = GPUDB_BGV;
    lastEntry.data.bigVal = 1;
    lastEntry.key = 10;
    lastEntry.parentID = 3;
    driver.create(lastEntry);

    Entry realLastEntry;
    realLastEntry.valType = GPUDB_BGV;
    realLastEntry.data.bigVal = 1;
    realLastEntry.key = 10;
    realLastEntry.parentID = 6;
    driver.create(realLastEntry);

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
    thrust::host_vector<CoreTupleType> * hostqueryResult = driver.getEntriesForFilterSet(filters);
    t2 = clock();
    float diff2 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device multi-filter query latency = %fms\n", diff2);

    return 0;

}