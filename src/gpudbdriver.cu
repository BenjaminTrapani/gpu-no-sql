//
// GPU DB Implementation
//

#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"
#include <time.h>
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/sequence.h"
#include "thrust/transform.h"

#include <cub/cub.cuh>

using namespace GPUDB;
typedef GPUDBDriver::CoreTupleType CoreTupleType;

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
    deviceIntermediateBuffer = new thrust::device_vector<CoreTupleType>(8);
}
GPUDBDriver::~GPUDBDriver(){
    delete deviceIntermediateBuffer;
    deviceIntermediateBuffer=0;
}

void GPUDBDriver::create(const CoreTupleType &object){
    deviceEntries.push_back(object);
}

struct IsPartialTupleMatch /*: thrust::unary_function<GPUDBDriver::CoreTupleType,bool>*/{
    typedef GPUDBDriver::CoreTupleType CoreTupleType;

    CUB_RUNTIME_FUNCTION __forceinline__
    IsPartialTupleMatch(const CoreTupleType & filter):_filter(filter){}

    CUB_RUNTIME_FUNCTION __forceinline__ __
    bool operator()(const CoreTupleType & val)const{
        //return true;
        return val == _filter;
    }

private:
    CoreTupleType _filter;
};

thrust::host_vector<CoreTupleType> GPUDBDriver::query(const CoreTupleType &searchFilter, const GPUSizeType limit){
    clock_t t1, t2;
    t1 = clock();
    void * tempStorage = NULL;
    size_t tempStorageSize = 0;
    size_t selectedCount = 0;
    thrust::device_ptr<CoreTupleType> entriesBegin = deviceEntries.data();
    thrust::device_ptr<CoreTupleType> intermediateBegin = deviceIntermediateBuffer->data();
    CoreTupleType * rawEntriesBegin = entriesBegin.get();
    CoreTupleType * rawIntermediateBegin = intermediateBegin.get();
    cub::DeviceSelect::If(tempStorage, tempStorageSize, rawEntriesBegin,
                          rawIntermediateBegin, &selectedCount, deviceEntries.size(),
                            IsPartialTupleMatch(searchFilter));
    /*thrust::copy_if(deviceEntries.begin(), deviceEntries.end(), deviceIntermediateBuffer->begin(),
                    IsPartialTupleMatch(searchFilter));*/
    t2 = clock();

    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device copy_if latency = %fms\n", diff);

    thrust::host_vector<CoreTupleType> resultVector(selectedCount);
    thrust::copy(deviceIntermediateBuffer->begin(), deviceIntermediateBuffer->begin()+selectedCount,
                    resultVector.begin());
    return resultVector;
}

thrust::host_vector<CoreTupleType> GPUDBDriver::queryOnHost(const CoreTupleType &searchFilter, const GPUSizeType limit){
    thrust::host_vector<CoreTupleType> hostEntries = deviceEntries;
    clock_t t1, t2;
    t1 = clock();
    thrust::copy_if(thrust::host, hostEntries.begin(), hostEntries.end(), deviceIntermediateBuffer->begin(),
                    IsPartialTupleMatch(searchFilter));
    t2 = clock();

    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("host copy_if latency = %fms\n", diff);

    return hostEntries;
}

void GPUDBDriver::update(const CoreTupleType &searchFilter, const CoreTupleType &updates){

}
void GPUDBDriver::deleteBy(const CoreTupleType &searchFilter){

}

void GPUDBDriver::sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter){
    //thrust::sort(deviceEntries.begin(), deviceEntries.end());
}

int main(int argc, char * argv[]){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));
    Entry entry1;
    entry1.key = 10;
    entry1.valType = GPUDB_INT;
    entry1.data.bigVal = 12;

    Entry entry2;
    entry2.key = 20;
    entry2.valType = GPUDB_INT;
    entry2.data.bigVal = 22;

    for(int i = 0; i < driver.getTableSize(); i++){
        CoreTupleType aTuple(10, entry1, entry1, entry2, entry2);
        driver.create(aTuple);
    }

    Entry entry3;
    entry3.key = 30;
    entry3.valType = GPUDB_INT;
    entry3.data.bigVal = 32;
    CoreTupleType specialTuple(10, Entry(), Entry(), Entry(), entry3);
    driver.create(specialTuple);

    CoreTupleType filterTuple(10, Entry(), Entry(), Entry(), entry3);

    clock_t t1, t2;
    t1 = clock();
    thrust::host_vector<CoreTupleType> queryResult = driver.query(filterTuple, 1);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    for(int i = 0; i < queryResult.size(); i++){
        int val = queryResult[i].get<4>().data.num;
        if(val) {
            printf("Iter data val = %i\n", val);
        }
    }

    printf("Query took %f milliseconds\n", diff);

    t1 = clock();
    thrust::host_vector<CoreTupleType> hostResult = driver.queryOnHost(filterTuple, 1);
    t2 = clock();
    float hostDiff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Host query took %f ms\n", hostDiff);
    return 0;

}
