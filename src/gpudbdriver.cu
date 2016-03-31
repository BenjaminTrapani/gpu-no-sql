//
// Created by Benjamin Trapani on 3/30/16.
//

#include "gpudbdriver.h"
#include <cuda.h>
#include "stdio.h"

void create(const DBElement * object){

}

//caller is responsible for freeing memory.
QueryResult * query(const DBElement * object);
void update(const DBElement * object);
void deleteEntry(const DBElement * object);
void sort(const DBElement * sortFilter, const DBElement * searchFilter);

int main(int argc, char * argv[]){
    int nDevices;
    const unsigned int gb = 1024 * 1024 * 1024;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  GPU Memory (GB): %f\n", ((float)prop.totalGlobalMem)/((float)(gb)));
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    }
}