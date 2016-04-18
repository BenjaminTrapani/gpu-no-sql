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

    struct InternalResult {
        DeviceVector_t * deviceResultPointer;
        size_t numItems;
        size_t beginOffset;

        InternalResult():deviceResultPointer(0), numItems(0), beginOffset(0){}
    };

    class GPUDBDriver {
    public:
        GPUDBDriver();
        ~GPUDBDriver();

        void create(const Doc & toCreate);
        void batchCreate(const std::vector<Doc> & docs);
        // Added for adding entries to docs - works in tandem with getDocumentID()
        void createEntries(const std::vector<Entry> entries);

        // TODO
        // needed functionality:
        // 1. Run the original filters to get the roots
        // 2. call getDocumentID with sourceFilters
        // 3. Remove all roots that do not match this parent ID - GPU Operation?
        // 4. Run get roots procedure as normal and return result
        std::vector<Doc> getDocumentsForFilterSet(const FilterSet & filters, const std::vector<FilterGroup> projectionFilters);

        void update(const Entry & searchFilter, const Entry &updates);

        void deleteBy(const Entry & searchFilter);

        inline size_t getTableSize() const {
            return numEntries;
        }

        inline size_t getNumEntries() const {
            return deviceEntries.size();
        }

        // TODO
        // return the id of the given filters applied in top down order, matching the key as in Additional Functionality 2
        // should error on any filters that do not fit the style
        //unsigned long long int getDocumentID(const FilterSet & sourceFilters);
        // dpne vis getDocumentsForFilterSet


        // TODO
        // Needed Additional Functionality

        // Note: Doc's will be stored with its given key in key and a special value in value that is unique
        // to Doc's - if the key matches, the value will as well. The type will be GPUDB_DOC

        // 1.
        // Make a way to match the key but not the value
        // Needed For: to do filters by key X when value doesn't matter, and getDocumentsForFilterSet Changes
        // Also needed for getDocumentID
        // Suggested Solution: When GPUDB_Type in a filter = GPUDB_ANY, do this



    private:
        size_t numEntries;
        DeviceVector_t deviceEntries;
        DeviceVector_t * deviceIntermediateBuffer1;
        DeviceVector_t * deviceIntermediateBuffer2;
        HostVector_t * hostResultBuffer;

        void create(const Entry &object);
        void searchEntries(const FilterGroup & filter, DeviceVector_t * resultsFromThisStage,
                            DeviceVector_t * resultsFromLastStage, const size_t numToSearch, size_t & numFound);
        InternalResult getRootsForFilterSet(const FilterSet & filters);
        void getEntriesForRoots(const InternalResult & rootResult, std::vector<Doc> & result);
        std::vector<Doc> getEntriesForRoots(const InternalResult & rootResults);

    };
}

#endif // SRC_GPUDBDRIVER_H