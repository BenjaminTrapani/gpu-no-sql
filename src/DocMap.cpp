//
// TODO
//

#include <vector>
#include <string>
#include "FilterSet.hpp"
#include "gpudbdriver.h"

using namespace GPUDB;

class DocMap {
public:
    DocMap(GPUDBDriver & d);
    // Returns the external id for the given doc path
    int addDoc(std::vector<std::string> strings);
    // returns the internal val for the external doc
    unsigned long long int getDoc(int docID);
    // return the paths vector
    std::vector<std::string> getPath(int docID);
    // Get the filter set to search for this doc
    unsigned long long int getFilterSet(int docID);
    // Removes the external id from the mappings and returns an exit code
    int removeDoc(int docID);
private:
    GPUDBDriver driver;
    std::vector<int> openSpots;
    unsigned long long int docs[1000];
    FilterSet filters[1000];
    std::vector<std::string> paths[1000];
};

DocMap::DocMap(GPUDBDriver & d) {
    // Set Up Open Docs
    openSpots.reserve(1000);
    for (int i = 999; i > 1; i--) {
        openSpots.push_back(i);
    }

    driver = driver;
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
    if (docID == 0) {
        return 0;
    }
    return docs[docID];
}

std::vector<std::string> DocMap::getPath(int docID) {
    if (docID == 0) {
        std::vector<std::string> empty;
        return empty;
    }
    return paths[docID];
}

FilterSet DocMap::getFilterSet(int docID) {
    if (docID == 0) {
        FilterSet empty;
        return empty;
    }
    return filters[docID];
}

int DocMap::removeDoc(int docID) {
    if (docID == 0) {
        return 0;
    }
    openSpots.push_back(docID);
    // TODO better add back with random swap

    return 0;
}


#endif //GPU_NO_SQL_DOCMAP_H
