//
// GPU DB Implementation
//

#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"
#include <time.h>

using namespace GPUDB;

__global__
void cudaQuery(Entry * data, Entry * filter, char * resultSet){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    resultSet[i] = data[i].key & filter->key & data[i].data.bigVal & filter->data.bigVal;
    /*if(data[i].key == filter->key && data[i].data.bigVal == filter->data.bigVal){

    }*/
}

GPUDBDriver::GPUDBDriver():nextFreeIndex(0){
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
    size_t allocSize = memBytes*0.15f;
    numEntries = allocSize/sizeof(Entry);

    //cudaMallocHost(&deviceMappedEntries, numEntries * sizeof(Entry));
    //cudaMallocHost(&deviceMappedFilter, sizeof(Entry));

    cudaMalloc(&deviceMappedEntries, numEntries * sizeof(Entry));
    cudaMalloc(&deviceMappedFilter, sizeof(Entry));

    cudaMalloc(&queryResults, numEntries * sizeof(char));
}
GPUDBDriver::~GPUDBDriver(){
    cudaFree(deviceMappedEntries);
    cudaFree(deviceMappedFilter);
    cudaFree(queryResults);
}

void GPUDBDriver::create(const GPUDB_Entry *object){
    cudaMemcpy(deviceMappedEntries+nextFreeIndex, object, sizeof(Entry), cudaMemcpyHostToDevice);
    nextFreeIndex++;
}
QueryResult GPUDBDriver::query(const GPUDB_Entry *searchFilter, const GPUSizeType limit){
    //*deviceMappedFilter = *searchFilter;

    int blockSize = 256;
    int blockCount = numEntries/blockSize;

    cudaQuery<<< blockCount, blockSize >>> (deviceMappedEntries, deviceMappedFilter, queryResults);

    QueryResult result(deviceMappedEntries, numEntries);
    return result;
}
void GPUDBDriver::update(const GPUDB_Entry *searchFilter, const GPUDB_Entry *updates){

}
void GPUDBDriver::deleteBy(const GPUDB_Entry *searchFilter){

}
void GPUDBDriver::sort(const GPUDB_Entry *sortFilter, const GPUDB_Entry *searchFilter){

}

int main(int argc, char * argv[]){
    GPUDBDriver driver;

    for(int i = 0; i < driver.getTableSize(); i++){
        GPUDB_Entry * entry = new GPUDB_Entry();
        entry->valType = GPUDB_INT;
        entry->data.bigVal = 4;
        entry->key = i;
        driver.create(entry);
    }

    GPUDB_Entry * filter = new GPUDB_Entry();
    filter->data.bigVal = 4;
    filter->key=driver.getTableSize()/2;

    clock_t beforeQuery;
    clock_t afterQuery;

    unsigned long long totalNanos = 0;
    for(int rounds = 0; rounds < 64; rounds++) {
        beforeQuery = clock();
        QueryResult queryResult = driver.query(filter, driver.getTableSize());
        afterQuery = clock();
        totalNanos += ((float)(afterQuery - beforeQuery) / 1000000.0F ) * 1000000000;
    }
    totalNanos/= 64;
    printf("Average query time = %llu\n", totalNanos);
    return 0;

}
