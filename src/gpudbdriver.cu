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
    size_t allocSize = memBytes*0.12f;
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

void GPUDBDriver::createSync(const Doc & toCreate) {
    create(toCreate.kvPair);
    for (std::vector<Doc>::const_iterator iter = toCreate.children.begin(); iter != toCreate.children.end(); ++iter) {
        create(*iter);
    }
    syncCreates();
}

void GPUDBDriver::batchCreate(std::vector<Doc> & docs) {
    for (std::vector<Doc>::iterator iter = docs.begin(); iter != docs.end(); ++iter){
        create(*iter);
    }
    syncCreates();
}

void GPUDBDriver::create(const Entry &object){
    hostCreateBuffer->push_back(object);
    //deviceEntries.push_back(object);
}

void GPUDBDriver::createEntries(const std::vector<Entry> entries) {
    for (std::vector<Entry>::const_iterator iter = entries.begin(); iter != entries.end(); ++iter) {
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

void GPUDBDriver::deleteBy(const Entry & searchFilter) {
    thrust::remove_if(deviceEntries.begin(), deviceEntries.end(), IsFullEntryMatch(searchFilter));
}


void GPUDBDriver::optimizedSearchEntries(const FilterGroup & filterGroup, const unsigned long int layer) {
    for (FilterGroup::const_iterator filterIter = filterGroup.group.begin(); filterIter != filterGroup.group.end();
         ++filterIter) {
        DeviceVector_t::iterator lastIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                            IsEntrySelected(layer));
        while (lastIter != deviceEntries.end()) {
            thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), SelectEntry(layer+1),
                                 FetchEntryWithParentID(*filterIter, thrust::raw_pointer_cast(&(*lastIter))));

            lastIter = thrust::find_if(lastIter+1, deviceEntries.end(), IsEntrySelected(layer));
        }
    }
}

InternalResult GPUDBDriver::optimizedGetRootsForFilterSet(const FilterSet & filters) {
    thrust::transform(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), UnselectEntry());

    for (FilterGroup::const_iterator firstGroupIter = filters[0].group.begin(); firstGroupIter != filters[0].group.end();
         ++firstGroupIter) {
        thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), SelectEntry(1),
                             IsPartialEntryMatch(*firstGroupIter));
    }

    unsigned long int layer = 1;
    for (FilterSet::const_iterator iter = filters.begin()+1; iter != filters.end(); ++iter) {
        optimizedSearchEntries(*iter, layer);
        layer++;
    }

    DeviceVector_t::iterator lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.end(),
                                                    deviceIntermediateBuffer1->begin(), IsEntrySelected(layer));

    InternalResult result;
    result.numItems = thrust::distance(deviceIntermediateBuffer1->begin(), lastIter);
    result.deviceResultPointer = deviceIntermediateBuffer1;
    result.beginOffset = 0;
    return result;
}

void GPUDBDriver::getEntriesForRoots(const InternalResult & rootResult, std::vector<Doc> & result) {
    DeviceVector_t::iterator lastIter;
    size_t numFound = 0;
    thrust::copy(rootResult.deviceResultPointer->begin() + rootResult.beginOffset,
                    rootResult.deviceResultPointer->begin() + rootResult.beginOffset + rootResult.numItems,
                 hostResultBuffer->begin() + rootResult.beginOffset);

    size_t iterIndex = 0;
    for (DeviceVector_t::const_iterator iter = rootResult.deviceResultPointer->begin() + rootResult.beginOffset;
        iter != rootResult.deviceResultPointer->begin() + rootResult.beginOffset + rootResult.numItems; ++iter) {

        Doc curDocVal((*hostResultBuffer)[rootResult.beginOffset + iterIndex]);
        result.push_back(curDocVal);

        DeviceVector_t::iterator destIter =
                rootResult.deviceResultPointer->begin() + rootResult.beginOffset + numFound + rootResult.numItems;
        lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.begin() + deviceEntries.size(),
                                   destIter,
                                   FetchDescendentEntry(thrust::raw_pointer_cast(&(*iter))));
        size_t mostRecentFoundCount = thrust::distance(destIter, lastIter);
        numFound += mostRecentFoundCount;

        if (mostRecentFoundCount) {
            InternalResult newQuery;
            newQuery.deviceResultPointer = rootResult.deviceResultPointer;
            newQuery.beginOffset = rootResult.beginOffset + numFound;
            newQuery.numItems = mostRecentFoundCount;
            getEntriesForRoots(newQuery, result[result.size() - 1].children);
        }

        iterIndex++;
    }
}

std::vector<Doc> GPUDBDriver::getEntriesForRoots(const InternalResult & rootResult) {
    std::vector<Doc> result;
    getEntriesForRoots(rootResult, result);

    return result;
}

// TODO new API Change, second argument currently ignored
std::vector<Doc> GPUDBDriver::getDocumentsForFilterSet(const FilterSet & filters) {
    clock_t t1, t2;
    t1 = clock();
    InternalResult rootResult = optimizedGetRootsForFilterSet(filters);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    //printf("get roots for filter set took %fms\n", diff);

    if (rootResult.numItems)
        return getEntriesForRoots(rootResult);

    return std::vector<Doc>(0);
}

// TODO stub
unsigned long long int GPUDBDriver::getDocumentID(const FilterSet & sourceFilters) {
    return 0; // TODO
}


