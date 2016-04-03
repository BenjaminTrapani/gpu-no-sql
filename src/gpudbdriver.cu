#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"
#include <time.h>
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/sequence.h"
#include "thrust/transform.h"

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
    size_t allocSize = memBytes*0.12f;
    numEntries = allocSize/sizeof(CoreTupleType);
    printf("Num entries = %i\n", numEntries);

    deviceEntries.reserve(numEntries);
    deviceIntermediateBuffer = new thrust::device_vector<CoreTupleType>(numEntries);
    deviceParentIndices = new thrust::device_vector<GPUSizeType>(numEntries);
    hostBuffer = new thrust::host_vector<GPUSizeType>(numEntries);
    hostResultBuffer = new thrust::host_vector<CoreTupleType>(numEntries);
}
GPUDBDriver::~GPUDBDriver(){
    delete deviceIntermediateBuffer;
    deviceIntermediateBuffer=0;

    delete deviceParentIndices;
    deviceParentIndices=0;

    delete hostBuffer;
    hostBuffer = 0;

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
    CoreTupleType _filter;
};

struct FetchParentID : thrust::unary_function<GPUDBDriver::CoreTupleType,GPUDBDriver::GPUSizeType>{
    __device__ __host__
    inline GPUDBDriver::GPUSizeType operator()(const CoreTupleType & ival)const{
        return ival.parentID;
    }
};

thrust::device_vector<CoreTupleType>* GPUDBDriver::query(const CoreTupleType &searchFilter, const GPUSizeType limit){
    clock_t t1, t2;
    t1 = clock();
    thrust::copy_if(deviceEntries.begin(), deviceEntries.end(), deviceIntermediateBuffer->begin(),
                    IsPartialTupleMatch(searchFilter));
    t2 = clock();

    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device copy_if latency = %fms\n", diff);
    return deviceIntermediateBuffer;
}


void GPUDBDriver::update(const CoreTupleType &searchFilter, const CoreTupleType &updates){

}
void GPUDBDriver::deleteBy(const CoreTupleType &searchFilter){

}

void GPUDBDriver::sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter){
}

thrust::host_vector<GPUSizeType> * GPUDBDriver::getParentIndicesForFilter(const CoreTupleType & searchFilter){
    thrust::device_vector<CoreTupleType>::iterator lastIter = thrust::copy_if(deviceEntries.begin(),
                                                                              deviceEntries.end(),
                                                                              deviceIntermediateBuffer->begin(),
                                                                              IsPartialTupleMatch(searchFilter));

    size_t numFound = thrust::distance(deviceIntermediateBuffer->begin(), lastIter);
    thrust::transform(deviceIntermediateBuffer->begin(), deviceIntermediateBuffer->begin() + numFound,
                      deviceParentIndices->begin(), FetchParentID());

    hostBuffer->resize(numFound);
    thrust::copy(deviceParentIndices->begin(), deviceParentIndices->begin()+numFound, hostBuffer->begin());
    return hostBuffer;
}

thrust::device_vector<GPUSizeType> * GPUDBDriver::getParentIndicesForFilterDevice(const CoreTupleType & searchFilter){
    thrust::device_vector<CoreTupleType>::iterator lastIter = thrust::copy_if(deviceEntries.begin(),
                                                                              deviceEntries.end(),
                                                                              deviceIntermediateBuffer->begin(),
                                                                              IsPartialTupleMatch(searchFilter));

    size_t numFound = thrust::distance(deviceIntermediateBuffer->begin(), lastIter);
    thrust::transform(deviceIntermediateBuffer->begin(), deviceIntermediateBuffer->begin() + numFound,
                      deviceParentIndices->begin(), FetchParentID());
    return deviceParentIndices;
}

thrust::host_vector<CoreTupleType> * GPUDBDriver::getEntriesForFilterSet(std::vector<CoreTupleType> filters){
    thrust::device_vector<GPUSizeType> * parentIndices = 0;
    for(std::vector<CoreTupleType>::iterator iter = filters.begin(); iter != filters.end(); ++iter){
        if(parentIndices){
            GPUSizeType val = static_cast<GPUSizeType>((*parentIndices)[0]);
            iter->id = val;
        }
        if(iter+1 == filters.end()){
            thrust::device_vector<CoreTupleType>::iterator lastIter = thrust::copy_if(deviceEntries.begin(),
                                                                                      deviceEntries.end(),
                                                                                      deviceIntermediateBuffer->begin(),
                                                                                      IsPartialTupleMatch(*iter));
            size_t numFound = thrust::distance(deviceIntermediateBuffer->begin(), lastIter);
            hostResultBuffer->resize(numFound);
            thrust::copy(deviceIntermediateBuffer->begin(), deviceIntermediateBuffer->begin()+numFound,
                         hostResultBuffer->begin());
            return hostResultBuffer;
        }
        parentIndices = getParentIndicesForFilterDevice(*iter);
    }
}

thrust::host_vector<GPUSizeType> * GPUDBDriver::getParentIndicesForFilterOnHost(const CoreTupleType & searchFilter){
    thrust::host_vector<CoreTupleType> hostEntries = deviceEntries;
    thrust::host_vector<CoreTupleType>::iterator lastIter = thrust::copy_if(thrust::host, hostEntries.begin(),
                                                                            hostEntries.end(),
                                                                            hostResultBuffer->begin(),
                                                                              IsPartialTupleMatch(searchFilter));

    size_t numFound = thrust::distance(hostResultBuffer->begin(), lastIter);

    thrust::transform(thrust::host, hostResultBuffer->begin(), hostResultBuffer->begin() + numFound,
                      hostBuffer->begin(), FetchParentID());

    return hostBuffer;
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

    clock_t t1, t2;
    t1 = clock();
    thrust::host_vector<GPUSizeType> * queryResult = driver.getParentIndicesForFilter(lastEntry);
    t2 = clock();

    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device query latency = %fms\n", diff);

    printf("Expect to find entries with parentID = 3, 6\n");
    for(int i = 0; i < queryResult->size(); i++){
        printf("Parent id = %llu \n", (*queryResult)[i]);
    }

    t1 = clock();
    thrust::host_vector<GPUSizeType> * hostqueryResult = driver.getParentIndicesForFilterOnHost(lastEntry);
    t2 = clock();
    float diff2 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("host query latency = %fms\n", diff2);

    Entry filter1 = realLastEntry;
    Entry filter2;
    filter2.data.bigVal=0;
    filter2.valType = GPUDB_BGV;
    filter2.key=realLastEntry.parentID;

    std::vector<Entry> filters;
    filters.push_back(filter1);
    filters.push_back(filter2);

    t1 = clock();
    thrust::host_vector<CoreTupleType> * multiFilterResult = driver.getEntriesForFilterSet(filters);
    t2 = clock();
    float diff3 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("multi-filter device query latency = %fms\n", diff3);

    printf("expect key=6, value=0\n");
    for(int i = 0; i < multiFilterResult->size(); i++){
        printf("Key=%llu Value=%llu \n", (*multiFilterResult)[i].key, (*multiFilterResult)[i].data.bigVal);
    }
    return 0;

}