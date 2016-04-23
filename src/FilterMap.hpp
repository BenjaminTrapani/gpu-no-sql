//
// header for FilterMap
//

#ifndef GPU_NO_SQL_FILTERMAP_H
#define GPU_NO_SQL_FILTERMAP_H

#include <list>
#include <string>
#include "FilterSet.hpp"
#include "gpudbdriver.hpp"
#include "presets.hpp"

using namespace GPUDB;

class FilterMap {
public:
    FilterMap();

    // Returns the external id for the given doc path
    int newFilter(FilterSet sourceDocFilter);

    // returns the internal val for the external doc
    FilterSet getFilter(int filterID);

    // adds the given filter group to the filter set ID
    int addToFilter(int filterID, Entry e, GPUDB_COMP comp);

    // advance filter one level
    int advanceFilter(int filterID);

    // Removes the external id from the mappings and returns an exit code
    int removeFilter(int filterID);
private:
    std::list<int> openSpots;

    FilterSet filters[MAX_RESOURCES];
    FilterGroup curGroups[MAX_RESOURCES];

    bool validID(int filterID);
};

#endif //GPU_NO_SQL_FILTERMAP_H
