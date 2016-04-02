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
}
GPUDBDriver::~GPUDBDriver(){
    delete deviceIntermediateBuffer;
    deviceIntermediateBuffer=0;

    delete hostBuffer;
    hostBuffer = 0;
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
    inline GPUDBDriver::GPUSizeType operator()(const CoreTupleType & ival, GPUDBDriver::GPUSizeType & oval)const{
        oval=ival.parentID;
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
                      deviceParentIndices->begin(), deviceParentIndices->begin()+numFound, FetchParentID());

    hostBuffer->resize(numFound);
    thrust::copy(deviceParentIndices->begin(), deviceParentIndices->begin()+numFound, hostBuffer->begin());
    return hostBuffer;
}

int main(int argc, char * argv[]){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));

    for(unsigned int i = 0; i < driver.getTableSize()-1; i++){
        Entry anEntry;
        anEntry.data.bigVal=0;
        driver.create(anEntry);
    }
    Entry lastEntry;
    lastEntry.valType = GPUDB_BGV;
    lastEntry.data.bigVal = 1;
    lastEntry.key = 10;
    lastEntry.parentID = 3;
    driver.create(lastEntry);

    clock_t t1, t2;
    t1 = clock();
    thrust::host_vector<GPUSizeType> * queryResult = driver.getParentIndicesForFilter(lastEntry);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device query latency = %fms\n", diff);

    for(int i = 0; i < queryResult->size(); i++){
        printf("Parent id = %llu \n", (*queryResult)[i]);
    }
    return 0;

}
