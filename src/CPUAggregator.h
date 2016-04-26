//
// Created by Benjamin Trapani on 4/26/16.
//

#ifndef GPU_NO_SQL_CPUAGGREGATOR_H
#define GPU_NO_SQL_CPUAGGREGATOR_H
#include "thrust/host_vector.h"
#include "Doc.hpp"
#include <unordered_map>
#include <list>
#include <vector>
#include "Entry.hpp"

namespace GPUDB{
    class CPUAggregator{
    public:
        void buildResults(const thrust::host_vector<Entry> & roots, const size_t numRoots,
                          std::vector<Doc> & results);
        void onEntryCreate(const Entry & toCreate);
        void onUpdate(const unsigned long long int id, const Entry & updatedVal);
        void onDelete(const unsigned long long int id);

        typedef std::unordered_map<unsigned long long int, Entry> IDEntryMap_t;
        typedef std::unordered_map<unsigned long long int, std::list<unsigned long long int>> IDToChildIDsMap_t;
    private:
        void buildResultsWithParent(Doc * parent);
        IDEntryMap_t idToEntryMap;
        IDToChildIDsMap_t idToChildIdsMap;
    };
}

#endif //GPU_NO_SQL_CPUAGGREGATOR_H
