//
// Represents a mapping of user ID's to the information needed to locate a doc
//

#include "DocMap.hpp"

using namespace GPUDB;

DocMap::DocMap() {
    // Empty
}

DocMap::DocMap(GPUDBDriver *d) {
    // Set Up Open Docs
    for (int i = 999; i > 0; i--) {
        openSpots.push_front(i);
    }

    // Set up root
    std::vector<std::string> emptyStrings;
    FilterSet emptyFilter;
    docs[0] = 0;
    paths[0] = emptyStrings;
    filters[0] = emptyFilter;

    // Set up driver pointer
    driver = d;
}

// On Success: Returns an int from 0 to MAX_RESOURCES-1
// Error -1 - No space to add another doc
// Error -5 - Invalid Key
// Error -9 - Bad Path
int DocMap::addDoc(std::vector <std::string> strings) {
    if (openSpots.size() == 0) {
        return -1; // Not enough resources
    }

    // Get a place
    int place = openSpots.front();
    openSpots.pop_front();

    // Create filter set
    FilterSet newFilterSet;
    for (int i = 0; i < strings.size(); i += 1) {
        FilterGroup g;

        // If Last, Set as the result set
        if (i == strings.size() - 1) {
            g.resultMember = true;
        } else {
            g.resultMember = false;
        }

        // Create the filter
        Filter newFilter;
        int res = StringConversion::stringToInt(newFilter.entry.key, strings.at(i));
        if (res != 0) {
            return -5; // Invalid Key
        }
        newFilter.entry.valType = GPUDB_DOC;

        newFilter.comparator = KEY_ONLY;

        // Add the entry to the group
        g.group.push_back(newFilter);

        // Add the group to the set
        newFilterSet.push_back(g);
    }

    int actualDocID = driver->getDocumentID(newFilterSet);
    if (actualDocID == 0) {
        removeDoc(place);
        return -9; // Bad Path
    }

    // Add it to filter set spot
    filters[place] = newFilterSet;
    // get documentID and add it to doc spot
    docs[place] = actualDocID;
    // add the path
    paths[place] = strings;
    // return the place
    return place;
}

// 0 - invalid doc reference
unsigned long long int DocMap::getDoc(int docID) {
    if (!validID(docID)) {
        return 0; // Invalid Doc Reference
    }
    return docs[docID];
}

// empty string = invalid doc reference
//       exception - when called on root (0)
std::vector<std::string> DocMap::getPath(int docID) {
    if (!validID(docID)) {
        return std::vector<std::string>(0); // invalid doc reference
    }
    return paths[docID];
}

// empty FilterSet = invalid doc reference
//       exception - when called on root (0)
FilterSet DocMap::getFilterSet(int docID) {
    if (!validID(docID)) {
        return FilterSet(); // invalid doc reference
    }
    return filters[docID];
}

// Returns 0 on success
// Error -2 - Invalid docID
// Error -4 - Cannot remove root reference
int DocMap::removeDoc(int docID) {
    if (!validID(docID)) {
        return -2; // Invalid docID
    }
    if (docID == 0) {
        return -1; // Cannot remove root
    }

    openSpots.push_back(docID);
    // TODO clear old data?

    return 0;
}

bool DocMap::validID(int filterID) {
    return filterID < MAX_RESOURCES && filterID >= 0; // TODO check if member?
}