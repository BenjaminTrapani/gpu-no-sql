//
// GPUDB Driver Header (API and Helpers)
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "DBStructs.h"
#include "thrust/tuple.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

// Caller must free memory
namespace GPUDB {
    class GPUDBDriver {
    public:
        typedef unsigned long long int GPUSizeType;
        typedef Entry CoreTupleType;
        typedef thrust::device_vector<CoreTupleType> DeviceVector_t;
        typedef thrust::host_vector<CoreTupleType> HostVector_t;

        GPUDBDriver();
        ~GPUDBDriver();
        void create(const CoreTupleType &object);

        thrust::device_vector<CoreTupleType>* query(const CoreTupleType &searchFilter, const GPUSizeType limit);

        thrust::host_vector<CoreTupleType> * getEntriesForFilterSet(std::vector<CoreTupleType> filters);

        void update(const CoreTupleType &searchFilter, const CoreTupleType &updates);
        void deleteBy(const CoreTupleType &searchFilter);
        void sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter);

        inline size_t getTableSize()const{
            return numEntries;
        }

    private:
        void searchEntries(const CoreTupleType & filter, DeviceVector_t * resultsFromThisStage,
                            DeviceVector_t * resultsFromLastStage,
                           const size_t numToSearch,
                           size_t &numFound);
        size_t numEntries;
        DeviceVector_t deviceEntries;
        DeviceVector_t * deviceIntermediateBuffer1;
        DeviceVector_t * deviceIntermediateBuffer2;
        HostVector_t * hostResultBuffer;
    };
}

#endif // SRC_GPUDBDRIVER_H