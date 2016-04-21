//
// TODO
//

#include "FilterMap.hpp"

FilterMap::FilterMap(GPUDBDriver *d, DocMap *m) {
    // Set Up Open Filters
    openSpots.reserve(1000);
    for (int i = 999; i > 1; i--) {
        openSpots.push_back(i);
    }
    // Initialize Pointers to Map / Driver
    documentMap = m;
    driver = d;
}

int FilterMap::newFilter(FilterSet sourceDocFilter) {
    return -1; // TODO
}

FilterSet FilterMap::getFilter(int filterID) {
    FilterSet s;
    return s; // TODO
}

int FilterMap::addToFilter(int filterID, Entry e) {
    return -1; // TODO
}

int FilterMap::advanceFilter(int filterID) {
    return -1; // TODO
}

int FilterMap::removeFilter(int filterID) {
    return -1; // TODO
}



