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

            GPUDBDriver();
            ~GPUDBDriver();
            void create(const CoreTupleType &object);

            thrust::device_vector<CoreTupleType>* query(const CoreTupleType &searchFilter, const GPUSizeType limit);
            //thrust::host_vector<CoreTupleType>* queryOnHost(const CoreTupleType &searchFilter, const GPUSizeType limit);

            thrust::host_vector<GPUSizeType> * getParentIndicesForFilter(const CoreTupleType & searchFilter);

            void update(const CoreTupleType &searchFilter, const CoreTupleType &updates);
            void deleteBy(const CoreTupleType &searchFilter);
            void sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter);

            inline size_t getTableSize()const{
                return numEntries;
            }

        private:
            size_t numEntries;
            thrust::device_vector<CoreTupleType> deviceEntries;
            thrust::device_vector<CoreTupleType> * deviceIntermediateBuffer;
            thrust::device_vector<GPUSizeType> * deviceParentIndices;
            thrust::host_vector<GPUSizeType> * hostBuffer;
    };
}

#endif // SRC_GPUDBDRIVER_H
