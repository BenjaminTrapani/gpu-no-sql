//
// User Level API Implementation
//

// Error Codes:
// 0 - success
// 1 - no space
// 2 - invalid doc reference
// 3 - lists are not of equal size
// 4 - cannot remove root
// 5 - Invalid Key
// 6 - invalid filter reference
// 7 - filter returns a document when it should not
// 9 - bad path to document

#include "gpudb-api.hpp"
#include "presets.hpp"
#include "StringConversion.hpp"
#include <cstdio>

using namespace GPUDB;

GPUDB_Database::GPUDB_Database():driver(), docs(&driver), filters(), curID(1) {
    // empty
}

// ********************************************************************************
// Document Identification

int GPUDB_Database::getRootDoc() {
    return 0;
}

// On Success: Returns an int from 0 to MAX_RESOURCES-1
// Errors: -1, -5, -9
int GPUDB_Database::getDocReference(std::vector<std::string> & strings) {
    return docs.addDoc(strings);
}

// Errors: 0, -2, -4
int GPUDB_Database::deleteDocRef(int docID) {
    return docs.removeDoc(docID);
}

// ********************************************************************************
// Creation

// Creates a document inside the given doc from ID with key "key"
// returns a new doc reference
// Errors: -1, -2, -5, -9
int GPUDB_Database::newDoc(int docID, std::string & key) {

    // Check for valid docID and create the new path
    std::vector<std::string> newPath = docs.getPath(docID);
    if (docID != 0 && newPath.empty()) {
        return -2; // Invalid Doc reference
    }
    newPath.push_back(key);

    // Create the actual new doc entry
    Entry newEntry;
    newEntry.parentID = docs.getDoc(docID);
    newEntry.id = curID;
    curID += 1;
    newEntry.valType = GPUDB_DOC;
    int res = StringConversion::stringToInt(newEntry.key, key);
    if (res != 0) {
        return -5; // Invalid Key
    }

    // Add New Entry to database
    driver.create(newEntry);
    driver.syncCreates();

    // Get a docID for the path
    int newDocID = getDocReference(newPath);
    if (newDocID < 0) {
        return newDocID; // bad path, no space, or invalid added key
    }

    // return the new doc ID
    return newDocID;
}

// adds to the given doc
// Errors: 0, -2, -5
int GPUDB_Database::addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {

    int realDocID = docs.getDoc(docID);
    if (realDocID == 0 && docID != 0) {
        return -2; // bad document reference
    }

    int res = addToDocNoSync(realDocID, key, value, type);
    driver.syncCreates();
    return res;
}

// adds to the given doc with syncing
// Errors: 0, -5
int GPUDB_Database::addToDocNoSync(unsigned long long int realDocID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {
    // Create a new Entry
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.parentID = realDocID;
    int res = StringConversion::stringToInt(newEntry.key, key);
    if (res != 0) {
        return -5; // bad key
    }
    newEntry.valType = type;
    setEntryVal(newEntry, value, type);

    driver.create(newEntry); // TODO add error code in driver?

    return 0;
}

void GPUDB_Database::setEntryVal(Entry & entry, GPUDB_Value & value, GPUDB_Type & type) {
    entry.data.bigVal = 0;
    if (type == GPUDB_BLN) {
        entry.data.b = value.b;
    } else if (type == GPUDB_INT) {
        entry.data.n = value.n;
    } else if (type == GPUDB_FLT) {
        entry.data.f = value.f;
    } else if (type == GPUDB_CHAR) {
        entry.data.c = value.c;
    } else if (type == GPUDB_STR) {
        memcpy(entry.data.s, value.s, sizeof(entry.data.s));
    } else {
        entry.data.bigVal = value.bigVal;
    }
}

// Adds in bulk
// Errors: 0, -2, -3
// -X - invalid key on the -x+1000000 place
long int GPUDB_Database::batchAdd(int docID, std::vector<std::string> & keys, std::vector<GPUDB_Value> & values, std::vector<GPUDB_Type> & types) {

    int realDocID = docs.getDoc(docID);
    if (realDocID == 0 && docID != 0) {
        return -2; // bad document reference
    }

    int keySize = keys.size();
    if (keySize != values.size() && keySize != types.size()) {
        return -3; // Lists are not of equal size
    }

    for (int i = 0; i < keySize; i += 1) {
        int res = addToDocNoSync(realDocID, keys.at(i), values.at(i), types.at(i));
        if (res != 0) {
            return -i - 1000000; // returns the negative of the value it failed on - all other values are not processed
        }
    }
    driver.syncCreates();
    return 0;
}

// ********************************************************************************
// Filter Creation

// Creates a new filter
// Errors: -1, -2
int GPUDB_Database::newFilter(int docID) {
    FilterSet docFilters = docs.getFilterSet(docID);
    if (docID != 0 && docFilters.empty()) {
        return -2; // Bad Document ID
    }
    return filters.newFilter(docFilters);
}

// Adds the given key to the filter
// Errors: -5, -6
int GPUDB_Database::addToFilter(int filterID, std::string key, GPUDB_Type & type) {

    // Translate key into an Entry for a search
    Entry newEntry;
    int res = StringConversion::stringToInt(newEntry.key, key);
    if (res != 0) {
        return -5; // Invalid Key
    }
    newEntry.valType = type;

    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, KEY_ONLY);
}

// Errors: -5, -6
int GPUDB_Database::addToFilter(int filterID, std::string key, GPUDB_Value & value, GPUDB_Type & type, GPUDB_COMP comp) {
    // Translate key and value into an Entry for a search
    Entry newEntry;
    int res = StringConversion::stringToInt(newEntry.key, key);
    if (res != 0) {
        return -5; // bad key
    }
    setEntryVal(newEntry, value, type);
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, comp);
}

// Errors: -6
int GPUDB_Database::advanceFilter(int filterID) {
    return filters.advanceFilter(filterID);
}

// Errors: -6
int GPUDB_Database::deleteFilter(int filterID) {
    return filters.removeFilter(filterID);
}

// ********************************************************************************
// Querying

// Errors:
GPUDB_QueryResult GPUDB_Database::query(int filterID) {
    FilterSet toFilter = filters.getFilter(filterID);
    //printf("FilterSet size: %d\n", toFilter.size());
    //printf("First Filter Group Size: %d\n", toFilter.front().group.size());
    clock_t t1, t2;
    float timeDiff;
    t1 = clock();
    std::vector<Doc> resultDoc = driver.getDocumentsForFilterSet(toFilter);
    t2 = clock();
    timeDiff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Time Taken For Query Internal: %fms.\n", timeDiff);
    if (resultDoc.size() != 1) {
        return GPUDB_QueryResult();
    }
    return translateDoc(resultDoc[0]);
}

GPUDB_QueryResult GPUDB_Database::translateDoc(Doc resultDoc) {
    // Set up the parent doc
    GPUDB_QueryResult userDoc;

    GPUDB_KV newKV;
    newKV.key = StringConversion::intToString(resultDoc.kvPair.key);
    newKV.type = resultDoc.kvPair.valType;
    if (newKV.type != GPUDB_DOC) {
        newKV.value = dataToValue(resultDoc.kvPair.data, newKV.type);
    }
    userDoc.kv = newKV;

    // Handle the children
    if (!resultDoc.children.empty()) {
        for (std::list<Doc>::iterator it = resultDoc.children.begin(); it != resultDoc.children.end(); it++) {
            GPUDB_QueryResult childResult = translateDoc(*it);
            userDoc.children.push_back(childResult);
        }
    }

    return userDoc;
}

GPUDB_Value GPUDB_Database::dataToValue(GPUDB_Data data, GPUDB_Type type) {
    GPUDB_Value v;
    v.bigVal = 0;
    if (type == GPUDB_BLN) {
        v.b = data.b;
    } else if (type == GPUDB_INT) {
        v.n = data.n;
    } else if (type == GPUDB_FLT) {
        v.f = data.f;
    } else if (type == GPUDB_CHAR) {
        v.c = data.c;
    } else if (type == GPUDB_STR) {
        memcpy(v.s, data.s, sizeof(data.s));
    } else {
        v.bigVal = data.bigVal;
    }
    return v;
}

// ********************************************************************************
// Updating

// Errors: -7
int GPUDB_Database::updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type & type) {
    // Get Matching Entry
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID)).front();

    // Check that it is not a doc
    Entry oldEntry = resultDoc.kvPair;
    if (oldEntry.valType == GPUDB_DOC) {
        return -7; // update cannot hit a document
    }

    // Create the revised entry
    Entry revisedEntry;
    revisedEntry.parentID = oldEntry.parentID;
    revisedEntry.id = oldEntry.id;
    memcpy(revisedEntry.key, oldEntry.key, sizeof(revisedEntry.key));
    revisedEntry.valType = type;
    setEntryVal(revisedEntry, value, type);

    // Update the entry
    driver.update(oldEntry, revisedEntry);

    return 0;
}

// ********************************************************************************
// Deleting

void GPUDB_Database::deleteFromDoc(int filterID) {
    // Get Matching Docs
    std::vector<Doc> resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID));
    // flatten docs into single vector
    std::list<Entry> allEntries;

    for (std::vector<Doc>::iterator it = resultDoc.begin(); it != resultDoc.end(); it++) {
        flattenDoc(*it, &allEntries);
    }


    for (std::list<Entry>::iterator it = allEntries.begin(); it != allEntries.end(); it++) {
        driver.deleteBy(*it);
    }

}

void GPUDB_Database::flattenDoc(Doc d, std::list<Entry> * targetEntryList) {
    targetEntryList->push_back(d.kvPair);
    if (!d.children.empty()) {
        for (std::list<Doc>::iterator it = d.children.begin(); it != d.children.end(); it++) {
            flattenDoc(*it, targetEntryList);
        }
    }
    return;
}

