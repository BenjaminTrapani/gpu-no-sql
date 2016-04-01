//
// GPUDB Driver Header (API and Helpers)
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "DBStructs.h"

// Caller must free memory
namespace GPUDB {
    class GPUDBDriver {
        public:
            typedef unsigned int GPUSizeType;

            GPUDBDriver();
            ~GPUDBDriver();
            void create(const GPUDB_Entry *object);
            QueryResult query(const GPUDB_Entry *searchFilter, const GPUSizeType limit);
            void update(const GPUDB_Entry *searchFilter, const GPUDB_Entry *updates);
            void deleteBy(const GPUDB_Entry *searchFilter);
            void sort(const GPUDB_Entry *sortFilter, const GPUDB_Entry *searchFilter);

            inline size_t getTableSize()const{
                return numEntries;
            }

        private:
            size_t numEntries;
            GPUSizeType nextFreeIndex;
            Entry * deviceMappedEntries;
            Entry * deviceMappedFilter;
            char * queryResults;
    };
}

#endif // SRC_GPUDBDRIVER_H
