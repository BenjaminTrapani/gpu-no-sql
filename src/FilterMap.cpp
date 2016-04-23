//
// Represents a
//

#include "FilterMap.hpp"

FilterMap::FilterMap(GPUDBDriver *d, DocMap *m) {
    // Set Up Open Filters
    for (int i = 999; i >= 0; i--) {
        openSpots.push_front(i);
    }
    // Initialize Pointers to Map / Driver
    //documentMap = m;
    //driver = d;
}

int FilterMap::newFilter(FilterSet sourceDocFilter) {
    int newID = openSpots.front();
    filters[newID] = sourceDocFilter;
    return newID;
}

FilterSet FilterMap::getFilter(int filterID) {
    FilterSet curFilters = filters[filterID];
    curFilters.push_back(curGroups[filterID]);
    return curFilters;
}

int FilterMap::addToFilter(int filterID, Entry e, GPUDB_COMP comp) {
    Filter newFilter;
    newFilter.entry = e;
    newFilter.comparator = comp;
    curGroups[filterID].group.push_back(newFilter);
}

int FilterMap::advanceFilter(int filterID) {
    curFilters.push_back(curGroups[filterID]);
    FilterGroup newGroup;
    curGroups[filterID] = newGroup;
    return 0;
}

int FilterMap::removeFilter(int filterID) {
    openSpots.push_back(docID);
    return 0;
}



