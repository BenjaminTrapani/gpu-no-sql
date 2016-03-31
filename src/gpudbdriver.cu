//
// GPU DB Implementation
//

#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"

int main(int argc, char * argv[]){
    int nDevices;
    const unsigned int gb = 1024 * 1024 * 1024;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  GPU Memory (GB): %f\n", ((float)prop.totalGlobalMem)/((float)(gb)));
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    }
}

void create(const GPUDB_Element *object) {
    return; // TODO
}

GPUDB_QueryResult * query(const GPUDB_Element *searchFilter) {
    return ((GPUDB_QueryResult *) malloc(sizeof(GPUDB_QueryResult)));
}

void update(const GPUDB_Element *searchFilter, const GPUDB_Element *updates) {
    return; // TODO
}

void deleteBy(const GPUDB_Element *searchFilter) {
    return; // TODO
}

void sort(const GPUDB_Element *sortFilter, const GPUDB_Element *searchFilter) {
    return; // TODO
}
