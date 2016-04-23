//
// GPUDB Driver Header (API and Helpers)
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "Entry.hpp"
#include "Doc.hpp"
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

        InternalResult():deviceResultPointer(0), numItems(0), beginOffset(0) {}
    };

    class GPUDBDriver {
    public:
        GPUDBDriver();
        ~GPUDBDriver();

        void create(const Doc & toCreate);
        void create(const Entry &object);
        void batchCreate(std::vector<Doc> & docs);
        void syncCreates();

        // TODO
        // needed functionality:
        // Search Top Down including the place of the first resultMember flag as the result
        // 1. Run the original filters to get the roots
        // 2. call getDocumentID with sourceFilters
        // 3. Remove all roots that do not match this parent ID - GPU Operation?
        // 4. Run get roots procedure as normal and return result
        std::vector<Doc> getDocumentsForFilterSet(const FilterSet & filters);

        void update(const Entry & searchFilter, const Entry &updates);

        void deleteBy(const Entry & searchFilter);

        inline size_t getTableSize() const {
            return numEntries;
        }

        inline size_t getNumEntries() const {
            return deviceEntries.size();
        }

        // return the id of the given filters applied in top down order, matching the key and that is a doc
        // should error on any filters that do not fit the style
        unsigned long long int getDocumentID(const FilterSet & sourceFilters);


        // TODO
        // Needed Additional Functionality

        // Note: Doc's will be stored with its given key in key and a special value in value that is unique
        // to Doc's - if the key matches, the value will as well. The type will be GPUDB_DOC

        // Make a way to match the key but not the value
        // Needed For: to do filters by key X when value doesn't matter, and getDocumentsForFilterSet Changes
        // Also needed for getDocumentID
        // Suggested Solution: When GPUDB_Type in a filter = GPUDB_ANY, do this

        // Make a comparator for value but not key

    private:
        size_t numEntries;
        DeviceVector_t deviceEntries;

        DeviceVector_t * deviceIntermediateBuffer1;
        HostVector_t * hostResultBuffer;
        HostVector_t * hostCreateBuffer;

        void optimizedSearchEntriesDown(const FilterGroup & filterGroup, const unsigned long int layer);
        unsigned long int selectAllSubelementsWithParentsSelected(const unsigned long int beginLayer);
        unsigned long int internalGetDocsForFilterSet(const FilterSet &filters);
        void buildResultsBottomUp(std::vector<Doc> & result, const unsigned long int beginLayer);

        std::vector<Doc> getEntriesForRoots(const InternalResult & rootResults);

    };
}

#endif // SRC_GPUDBDRIVER_H