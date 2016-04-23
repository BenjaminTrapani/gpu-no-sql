//
// Represents a mapping of user ID's to the information needed to locate a doc
//

#include "DocMap.hpp"

using namespace GPUDB;

DocMap::DocMap(GPUDBDriver *d) {
    // Set Up Open Docs
    for (int i = 999; i > 0; i--) {
        openSpots.push_front(i);
    }

    // Set up root
    std::vector<std::string> empty;
    FilterSet empty;
    docs[0] = 0;
    paths[0] = empty;
    filters[0] = empty;

    // Set up driver pointer
    driver = d;
}

int DocMap::addDoc(std::vector <std::string> strings) {
    if (openSpots.size() == 0) {
        return -1; // TODO error code
    }

    // Get a place
    int place = openSpots.front();
    openSpots.pop_front();

    // Create filter set
    FilterSet filters;
    for (int i = 0; i < strings.size(); i += 1) {
        FilterGroup g;

        // Create the entry
        Entry newEntry;

        newEntry.valType = GPUDB_DOC;
        if (strings.at(i).size() < 16) {
            newEntry.key = stringToInt(strings.at(i).c_str());
        } else {
            return -1; // TODO error code
        }

        // Create the filter
        Filter newFilter;
        newFilter.entry = newEntry;
        newFilter.comparator = KEY_ONLY;

        // Add the entry to the group
        g.group.push_back(newFilter);

        // If Last, Set as the result set
        if (i == strings.size() - 1) {
            g.resultSet = true;
        } else {
            g.resultSet = false;
        }

        // Add the group to the set
        filters.push_back(g);
    }

    // Add it to filter set spot
    docs[place] = filters;
    // get documentID and add it to doc spot
    docs[place] = driver.getDocumentID(filters);
    // return the place
    return place;
}

unsigned long long int DocMap::getDoc(int docID) {
    if (!validID(docID)) {
        return -1; // TODO error code
    }
    return docs[docID];
}

std::vector<std::string> DocMap::getPath(int docID) {
    if (!validID(docID)) {
        return -1; // TODO error code
    }
    return paths[docID];
}

FilterSet DocMap::getFilterSet(int docID) {
    if (!validID(docID)) {
        return -1; // TODO error code
    }
    return filters[docID];
}

int DocMap::removeDoc(int docID) {
    if (!validID(docID)) {
        return -1; // TODO error code
    }
    if (docID == 0) {
        return -1; // TODO error code, cannot remove root
    }
    openSpots.push_back(docID);

    return 0;
}

bool DocMap::validID(int filterID) {
    return filterID < maxResources && filterID >= 0;
}