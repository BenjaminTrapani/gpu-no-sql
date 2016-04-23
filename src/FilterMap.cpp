//
// Represents the filters currently in use and available filter spots
//

#include "FilterMap.hpp"

FilterMap::FilterMap() {
    // Set Up Open Filters
    for (int i = 999; i >= 0; i--) {
        openSpots.push_front(i);
    }
}

int FilterMap::newFilter(FilterSet sourceDocFilter) {
    int newID = openSpots.front();
    filters[newID] = sourceDocFilter;
    return newID;
}

FilterSet FilterMap::getFilter(int filterID) {
    if (!validID(filterID)) {
        FilterSet empty;
        return empty; // TODO error code
    }
    FilterSet curFilters = filters[filterID];
    curFilters.push_back(curGroups[filterID]);
    return curFilters;
}

int FilterMap::addToFilter(int filterID, Entry e, GPUDB_COMP comp) {
    if (!validID(filterID)) {
        return -1; // TODO error code
    }
    Filter newFilter;
    newFilter.entry = e;
    newFilter.comparator = comp;
    curGroups[filterID].group.push_back(newFilter);
    return 0;
}

int FilterMap::advanceFilter(int filterID) {
    if (!validID(filterID)) {
        return -1; // TODO error code
    }
    curFilters.push_back(curGroups[filterID]);
    FilterGroup newGroup;
    curGroups[filterID] = newGroup;
    return 0;
}

int FilterMap::removeFilter(int filterID) {
    if (!validID(filterID)) {
        return -1; // TODO error code
    }
    openSpots.push_back(docID);
    return 0;
}

bool FilterMap::validID(int filterID) {
    return filterID < maxResources && filterID >= 0;
}



