//
// GPUDB Driver Header (API and Helpers)
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "Entry.h"
#include "Doc.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "FilterSet.hpp"

// Caller must free memory
namespace GPUDB {
    typedef unsigned long long int GPUSizeType;
    typedef thrust::device_vector<Entry> DeviceVector_t;
    typedef thrust::host_vector<Entry> HostVector_t;

    struct QueryResult{
        DeviceVector_t * deviceResultPointer;
        size_t numItems;
        size_t beginOffset;

        QueryResult():deviceResultPointer(0), numItems(0), beginOffset(0){}
    };

    class GPUDBDriver {
    public:
        GPUDBDriver();
        ~GPUDBDriver();

        void create(const Doc & toCreate);
        void batchCreate(const std::vector<Doc> & docs);

        std::vector<Doc> getDocumentsForFilterSet(const FilterSet& filters);

        void update(const Entry &searchFilter, const Entry &updates);

        void deleteBy(const Entry &searchFilter);

        inline size_t getTableSize()const{
            return numEntries;
        }
        inline size_t getNumEntries()const{
            return deviceEntries.size();
        }

    private:
        void create(const Entry &object);
        void searchEntries(const FilterGroup & filter, DeviceVector_t * resultsFromThisStage,
                            DeviceVector_t * resultsFromLastStage,
                           const size_t numToSearch,
                           size_t &numFound);
        QueryResult getRootsForFilterSet(const FilterSet& filters);
        void getEntriesForRoots(const QueryResult & rootResult, std::vector<Doc> & result);
        std::vector<Doc> getEntriesForRoots(const QueryResult & rootResults);

        size_t numEntries;
        DeviceVector_t deviceEntries;
        DeviceVector_t * deviceIntermediateBuffer1;
        DeviceVector_t * deviceIntermediateBuffer2;
        HostVector_t * hostResultBuffer;
    };
}

#endif // SRC_GPUDBDRIVER_H