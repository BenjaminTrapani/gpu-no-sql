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

// Creates a new filter - returns -1 if there isn't enough resoruces
int FilterMap::newFilter(FilterSet sourceDocFilter) {
    if (openSpots.empty()) {
        return -1;
    }

    int newID = openSpots.front();
    filters[newID] = sourceDocFilter;
    return newID;
}

FilterSet FilterMap::getFilter(int filterID) {
    if (!validID(filterID)) {
        FilterSet empty;
        return empty; // Invalid filter reference
    }
    FilterSet curFilters = filters[filterID];
    if (curGroups[filterID].group.size() != 0) {
        curFilters.push_back(curGroups[filterID]);
    }
    return curFilters;
}

// -6 - invalid filter reference
int FilterMap::addToFilter(int filterID, Entry e, GPUDB_COMP comp) {
    if (!validID(filterID)) {
        return -6; // invalid filter reference
    }
    Filter newFilter;
    newFilter.entry = e;
    newFilter.comparator = comp;
    curGroups[filterID].group.push_back(newFilter);
    return 0;
}

// -6 - invalid filter reference
int FilterMap::advanceFilter(int filterID) {
    if (!validID(filterID)) {
        return -6; // invalid filter reference
    }
    filters[filterID].push_back(curGroups[filterID]);
    FilterGroup newGroup;
    curGroups[filterID] = newGroup;
    return 0;
}

int FilterMap::removeFilter(int filterID) {
    if (!validID(filterID)) {
        return -6; // invalid filter reference
    }
    openSpots.push_back(filterID);
    // TODO clear old data?
    return 0;
}

bool FilterMap::validID(int filterID) {
    return filterID < MAX_RESOURCES && filterID >= 0;
}



