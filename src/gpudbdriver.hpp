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