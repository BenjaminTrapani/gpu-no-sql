//
// Represents a mapping of user ID's to the information needed to locate a doc
//

#include "DocMap.hpp"

using namespace GPUDB;

DocMap::DocMap(GPUDBDriver *d) {
    // Set Up Open Docs
    openSpots.reserve(1000);
    for (int i = 999; i > 1; i--) {
        openSpots.push_back(i);
    }

    // Set up root
    std::vector<std::string> empty;
    FilterSet empty;
    docs[0] = 0;
    paths[0] = empty;
    filters[0] = empty;

    driver = d;
}

int DocMap::addDoc(std::vector <std::string> strings) {
    if (openSpots.size() == 0) {
        return -1; // TODO error code
    }

    // Get a place
    int place = openSpots.back();
    openSpots.pop_back();

    // Create filter set
    FilterSet filters;
    for (int i = 0; i < strings.size(); i += 1) {
        FilterGroup g;

        // Create the entry
        Entry newFilter;

        // TODO set filter type with expansion accordingly

        newFilter.valType = GPUDB_DOC;
        if (strings.at(i).size() < 16) {
            newFilter.key = strings.at(i).c_str();
        } else {
            return -1; // TODO error code
        }

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
    return docs[docID];
}

std::vector<std::string> DocMap::getPath(int docID) {
    return paths[docID];
}

FilterSet DocMap::getFilterSet(int docID) {
    return filters[docID];
}

int DocMap::removeDoc(int docID) {
    if (docID == 0) {
        return -1; // TODO error code, cannot remove root
    }
    openSpots.push_back(docID);
    // TODO better add back with random swap

    return 0;
}