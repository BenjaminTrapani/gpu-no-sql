//
// TODO
//

#ifndef GPU_NO_SQL_FILTERMAP_H
#define GPU_NO_SQL_FILTERMAP_H

#include <vector>
#include <string>
#include "FilterSet.hpp"
#include "gpudbdriver.hpp"

using namespace GPUDB;

class FilterMap {
public:
    FilterMap(GPUDBDriver *d, DocMap *m);

    // Returns the external id for the given doc path
    int newFilter(FilterSet sourceDocFilter);

    // returns the internal val for the external doc
    FilterSet getFilter(int filterID);

    // adds the given filter group to the filter set ID
    int addToFilter(int filterID, Entry e);

    // advance filter one level
    int advanceFilter(int filterID);

    // Removes the external id from the mappings and returns an exit code
    int removeFilter(int filterID);
private:
    GPUDBDriver *driver;
    DocMap *documentMap;

    std::vector<int> openSpots; // TODO switch to list

    FilterSet filters[1000];
    FilterGroup curGroups[1000];
};

#endif //GPU_NO_SQL_FILTERMAP_H
