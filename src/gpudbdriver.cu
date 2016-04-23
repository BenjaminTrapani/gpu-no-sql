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
    size_t allocSize = memBytes*0.25f; //0.12
    numEntries = allocSize/sizeof(Entry);
    printf("Num entries = %i\n", numEntries);

    //buffer allocation and initialization
    deviceEntries.reserve(numEntries);

    //deviceIntermediateBuffer1 = new DeviceVector_t(numEntries);
    hostResultBuffer = new HostVector_t(numEntries);
    hostCreateBuffer = new HostVector_t();
    hostCreateBuffer->reserve(numEntries);
}

GPUDBDriver::~GPUDBDriver() {
    //delete deviceIntermediateBuffer1;
    //deviceIntermediateBuffer1=0;

    delete hostResultBuffer;
    hostResultBuffer=0;

    delete hostCreateBuffer;
    hostCreateBuffer = 0;
}

void GPUDBDriver::create(const Doc & toCreate) {
    create(toCreate.kvPair);
    for (std::list<Doc>::const_iterator iter = toCreate.children.begin(); iter != toCreate.children.end(); ++iter) {
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

void GPUDBDriver::deleteBy(const Entry & searchFilter) {
    thrust::remove_if(deviceEntries.begin(), deviceEntries.end(), IsFullEntryMatch(searchFilter));
}


void GPUDBDriver::optimizedSearchEntriesDown(const FilterGroup & filterGroup, const unsigned long int layer) {
    for (FilterGroup::const_iterator filterIter = filterGroup.group.begin(); filterIter != filterGroup.group.end();
         ++filterIter) {
        DeviceVector_t::iterator lastIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                            IsEntrySelected(layer));
        while (lastIter != deviceEntries.end()) {
            switch (filterIter->comparator){
                case GREATER:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsEntryGreater>(IsEntryGreater(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case GREATER_EQ:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsEntryGreaterEQ>(IsEntryGreaterEQ(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case EQ:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsPartialEntryMatch>(IsPartialEntryMatch(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case LESS_EQ:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsEntryLessEQ>(IsEntryLessEQ(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case LESS:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsEntryLess>(IsEntryLess(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case KEY_ONLY:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<EntryKeyMatch>(EntryKeyMatch(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                case VAL_ONLY:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<EntryValMatch>(EntryValMatch(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                    break;
                }
                default:{
                    thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                         SelectEntry(layer+1, filterGroup.resultMember),
                                         FetchEntryWithChildID<IsPartialEntryMatch>(IsPartialEntryMatch(filterIter->entry),
                                                               thrust::raw_pointer_cast(&(*lastIter))
                                         ));
                }
            }

            lastIter = thrust::find_if(lastIter+1, deviceEntries.end(), IsEntrySelected(layer));
        }
    }
}

unsigned long int GPUDBDriver::selectAllSubelementsWithParentsSelected(const unsigned long int beginLayer){
    unsigned long int curLayer = beginLayer;
    bool foundOneOnLayer = true;

    while(foundOneOnLayer) {
        foundOneOnLayer = false;
        DeviceVector_t::iterator lastIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                            IsEntrySelected(curLayer));
        while (lastIter != deviceEntries.end()) {
            thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                 SelectEntry(curLayer + 1, false),
                                 GetElementWithParent(thrust::raw_pointer_cast(&(*lastIter))));
            lastIter = thrust::find_if(lastIter + 1, deviceEntries.end(),
                                       IsEntrySelected(curLayer));
            foundOneOnLayer = true;
        }
        if(foundOneOnLayer)
            curLayer++;
    }

    return curLayer-1;
}

unsigned long int GPUDBDriver::internalGetDocsForFilterSet(const FilterSet &filters) {
    clock_t t1, t2;
    t1 = clock();
    thrust::transform(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(), UnselectEntry());
    for (FilterGroup::const_iterator firstGroupIter = filters[0].group.begin();
         firstGroupIter != filters[0].group.end();
         ++firstGroupIter) {
        switch (firstGroupIter->comparator){
            case GREATER:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     IsEntryGreater(firstGroupIter->entry));
                break;
            }
            case GREATER_EQ:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     IsEntryGreaterEQ(firstGroupIter->entry));
                break;
            }
            case EQ:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     IsPartialEntryMatch(firstGroupIter->entry));
                break;
            }
            case LESS_EQ:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     IsEntryLessEQ(firstGroupIter->entry));
                break;
            }
            case LESS:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     IsEntryLess(firstGroupIter->entry));
                break;
            }
            case KEY_ONLY:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     EntryKeyMatch(firstGroupIter->entry));
                break;
            }
            case VAL_ONLY:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                     SelectEntry(1, filters[0].resultMember),
                                     EntryValMatch(firstGroupIter->entry));
                break;
            }
            default:{
                thrust::transform_if(deviceEntries.begin(), deviceEntries.end(), deviceEntries.begin(),
                                            SelectEntry(1, filters[0].resultMember),
                                            IsPartialEntryMatch(firstGroupIter->entry));
            }
        }
    }

    unsigned long int layer = 1;
    for (FilterSet::const_iterator iter = filters.begin()+1; iter != filters.end(); ++iter) {
        optimizedSearchEntriesDown(*iter, layer);
        layer++;
    }
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("    Selecting all for querry took %fms\n", diff);

    t1 = clock();
    layer = selectAllSubelementsWithParentsSelected(layer);
    t2 = clock();
    diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("    Selecting all subelements took %fms\n", diff);

    return layer;
}

void GPUDBDriver::buildResultsBottomUp(std::vector<Doc> & result, const unsigned long int beginLayer){
    /*DeviceVector_t::iterator lastIter = thrust::copy_if(deviceEntries.begin(), deviceEntries.end(),
                                                        deviceIntermediateBuffer1->begin(),
                                                        IsEntrySelected(beginLayer));*/

    std::unordered_map<unsigned long int, Doc> docIDMap;
    DeviceVector_t::iterator iter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                    IsEntrySelected(beginLayer));
    while (iter != deviceEntries.end()){

        DeviceVector_t::iterator childIter = iter;
        Entry curHostChild = *childIter;
        docIDMap[curHostChild.id] = Doc(curHostChild);
        DeviceVector_t::iterator parentIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                                              GetElementWithChild(thrust::raw_pointer_cast(&(*childIter))));

        Doc * lastValidParent;
        if (curHostChild.isResultMember) {
            lastValidParent = &docIDMap[curHostChild.id];
        } else {
            lastValidParent = 0;
        }
        while (parentIter != deviceEntries.end()) {
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
                break;
            }
            parentIter = thrust::find_if(deviceEntries.begin(), deviceEntries.end(),
                                         GetElementWithChild(thrust::raw_pointer_cast(&(*parentIter))));

        }
        if(lastValidParent) {
            result.push_back(*lastValidParent);
        }
        iter = thrust::find_if(iter + 1, deviceEntries.end(),
                               IsEntrySelected(beginLayer));
    }
}

std::vector<Doc> GPUDBDriver::getDocumentsForFilterSet(const FilterSet & filters) {
    clock_t t1, t2;
    t1 = clock();
    unsigned long int finalLevel = internalGetDocsForFilterSet(filters);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("  Select matches took %fms\n", diff);
    std::vector<Doc> result;
    t1 = clock();
    buildResultsBottomUp(result, finalLevel);
    t2 = clock();
    diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("  Build results took %fms\n", diff);
    return result;
}

unsigned long long int GPUDBDriver::getDocumentID(const FilterSet & sourceFilters) {
    std::vector<Doc> result = getDocumentsForFilterSet(sourceFilters);

    if(result.size()==1)
        return getDocumentsForFilterSet(sourceFilters)[0].kvPair.id;

    return 0;
}


