#include "gpudbdriver.hpp"
#include "Functors.hpp"

#include <cuda.h>
#include "stdio.h"
#include <time.h>
#include "thrust/sort.h"
#include "thrust/copy.h"
#include "thrust/execution_policy.h"
#include "thrust/sequence.h"
#include "thrust/transform.h"
#include "thrust/device_ptr.h"
#include <unordered_map>

using namespace GPUDB;

GPUDBDriver::GPUDBDriver() {
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
    size_t allocSize = memBytes*0.08f; //0.12
    numEntries = allocSize/sizeof(Entry);
    printf("Num entries = %i\n", numEntries);

    //buffer allocation and initialization
    deviceEntries.reserve(numEntries);

    deviceIntermediateBuffer1 = new DeviceVector_t(numEntries);
    hostResultBuffer = new HostVector_t(numEntries);
    hostCreateBuffer = new HostVector_t();
    hostCreateBuffer->reserve(numEntries);
}

GPUDBDriver::~GPUDBDriver() {
    delete deviceIntermediateBuffer1;
    deviceIntermediateBuffer1=0;

    delete hostResultBuffer;
    hostResultBuffer=0;

    delete hostCreateBuffer;
    hostCreateBuffer = 0;
}

void GPUDBDriver::create(const Doc & toCreate) {
    create(toCreate.kvPair);
    for (std::vector<Doc>::const_iterator iter = toCreate.children.begin(); iter != toCreate.children.end(); ++iter) {
        create(*iter);
    }
}

void GPUDBDriver::create(const Entry &object){
    hostCreateBuffer->push_back(object);
    //deviceEntries.push_back(object);
}

void GPUDBDriver::batchCreate(std::vector<Doc> & docs) {
    for (std::vector<Doc>::iterator iter = docs.begin(); iter != docs.end(); ++iter){
        create(*iter);
    }
    syncCreates();
}

void GPUDBDriver::syncCreates() {
    DeviceVector_t::iterator oldEnd = deviceEntries.end();
    deviceEntries.resize(deviceEntries.size() + hostCreateBuffer->size());
    thrust::copy(hostCreateBuffer->begin(), hostCreateBuffer->end(), oldEnd);
    hostCreateBuffer->clear();
}

void GPUDBDriver::update(const Entry & searchFilter, const Entry & updates) {
    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), ModifyEntry(updates),
                         IsFullEntryMatch(searchFilter));
}

void GPUDBDriver::deleteAll(const Entry & searchFilter) {
    thrust::remove_if(deviceEntries.begin(), deviceEntries.end(), IsFullEntryMatch(searchFilter));
}


void GPUDBDriver::optimizedSearchEntriesDown(const FilterGroup & filterGroup, const unsigned long int layer) {
    for (FilterGroup::const_iterator filterIter = filterGroup.group.begin(); filterIter != filterGroup.group.end();
         ++filterIter) {
        DeviceVector_t::iterator lastIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                            IsEntrySelected(layer));
        while (lastIter != deviceEntries.end()) {
            thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                 SelectEntry(layer-1, filterGroup.resultMember),
                                 FetchEntryWithChildID(*filterIter, thrust::raw_pointer_cast(&(*lastIter))));

            lastIter = thrust::find_if(lastIter+1, deviceEntries.end(), IsEntrySelected(layer));
        }
    }
}

unsigned long int GPUDBDriver::internalGetDocsForFilterSet(const FilterSet &filters) {
    thrust::transform(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), UnselectEntry());
    for (FilterGroup::const_iterator firstGroupIter = filters[0].group.begin(); firstGroupIter != filters[0].group.end();
         ++firstGroupIter) {
        thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                             SelectEntry(filters.size(), filters[0].resultMember),
                             IsPartialEntryMatch(*firstGroupIter));
    }

    unsigned long int layer = filters.size();
    for (FilterSet::const_iterator iter = filters.begin()+1; iter != filters.end(); ++iter) {
        optimizedSearchEntriesDown(*iter, layer);
        layer--;
    }

    return layer;
}

void GPUDBDriver::buildResultsBottomUp(std::vector<Doc> & result, const unsigned long int beginLayer){
    DeviceVector_t::iterator lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.end(),
                                                        deviceIntermediateBuffer1->begin(),
                                                        IsEntrySelected(beginLayer));

    std::unordered_map<unsigned long int, Doc> docIDMap;
    for(DeviceVector_t::iterator iter = deviceIntermediateBuffer1->begin();
            iter != lastIter; ++iter){

        DeviceVector_t::iterator childIter = iter;
        Entry curHostChild = *childIter;
        docIDMap[curHostChild.id] = Doc(curHostChild);
        DeviceVector_t::iterator parentIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                              GetElementWithChild(thrust::raw_pointer_cast(&(*childIter))));

        Doc * lastValidParent;
        if(curHostChild.isResultMember){
            lastValidParent = &docIDMap[curHostChild.id];
        }else{
            lastValidParent = 0;
        }
        while(parentIter != deviceEntries.end()){

            Entry hostChild = *childIter;
            Entry hostParent = *parentIter;

            std::unordered_map<unsigned long int, Doc>::iterator keyIndex = docIDMap.find(hostParent.id);
            if(keyIndex == docIDMap.end()){
                docIDMap[hostParent.id] = Doc(hostParent);
            }

            docIDMap[hostParent.id].addChild(docIDMap[hostChild.id]);

            childIter = parentIter;

            if(hostParent.isResultMember){
                lastValidParent = &docIDMap[hostParent.id];
            }
            parentIter = thrust::find_if(parentIter + 1, deviceEntries.end(),
                                         GetElementWithChild(thrust::raw_pointer_cast(&(*parentIter))));

        }
        if(lastValidParent) {
            result.push_back(*lastValidParent);
        }
    }
}

std::vector<Doc> GPUDBDriver::getDocumentsForFilterSet(const FilterSet & filters) {
    clock_t t1, t2;
    t1 = clock();
    unsigned long int finalLevel = internalGetDocsForFilterSet(filters);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;

    std::vector<Doc> result;
    buildResultsBottomUp(result, finalLevel);
    return result;
}

unsigned long long int GPUDBDriver::getDocumentID(const FilterSet & sourceFilters) {
    std::vector<Doc> result = getDocumentsForFilterSet(sourceFilters);

    if(result.size()==1)
        return getDocumentsForFilterSet(sourceFilters)[0].kvPair.id;

    return 0;
}


