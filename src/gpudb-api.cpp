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
#include "stdio.h"
#include <cstdio>
#include "presets.hpp"
#include "StringConversion.hpp"

using namespace GPUDB;

GPUDB::GPUDB():driver(), docs(&driver), filters(), curID(0) {
    // empty
}

// ********************************************************************************
// Document Identification

int GPUDB::getRootDoc() {
    return 0;
}

// On Success: Returns an int from 0 to MAX_RESOURCES-1
// Errors: -1, -5, -9
int GPUDB::getDoc(std::vector<std::string> & strings) {
    return docs.addDoc(strings);
}

// Errors: 0, -2, -4
int GPUDB::deleteDocRef(int docID) {
    return docs.removeDoc(docID);
}

// ********************************************************************************
// Creation

// Creates a document inside the given doc from ID with key "key"
// Errors: -1, -2, -5, -9
int GPUDB::newDoc(int docID, std::string & key) {
    // Create the new path
    std::vector<std::string> newPath = docs.getPath(docID);
    if (docID != 0 && newPath.empty()) {
        return -2; // Invalid Doc reference
    }
    newPath.push_back(key);

    // Get a docID for the path
    int newDocID = getDoc(newPath);
    if (newDocID < 0) {
        return newDocID; // bad path, no space, or invalid added key
    }

    // Create the actual new doc entry
    unsigned long long int parentID = docs.getDoc(docID);
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.valType = GPUDB_DOC;
    StringConversion::stringToInt(newEntry.key, key);

    // Add New Entry to database
    driver.create(newEntry);
    driver.syncCreates();

    // return the new doc ID
    return newDocID;
}

// adds to the given doc
// Errors: 0, -2, -5
int GPUDB::addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {

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
int GPUDB::addToDocNoSync(unsigned long long int realDocID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {
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

void GPUDB::setEntryVal(Entry & entry, GPUDB_Value & value, GPUDB_Type & type) {
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
long int GPUDB::batchAdd(int docID, std::vector<std::string> & keys, std::vector<GPUDB_Value> & values, std::vector<GPUDB_Type> & types) {

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
int GPUDB::newFilter(int docID) {
    FilterSet docFilters = docs.getFilterSet(docID);
    if (docID != 0 && docFilters.empty()) {
        return -2; // Bad Document ID
    }
    return filters.newFilter();
}

// Adds the given key to the filter
// Errors: -5, -6
int GPUDB::addToFilter(int filterID, std::string key) {

    // Translate key into an Entry for a search
    Entry newEntry;
    int res = StringConversion::stringToInt(newEntry.key, key);
    if (res != 0) {
        return -5; // Invalid Key
    }
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, KEY_ONLY);
}

// Errors: -5, -6
int GPUDB::addToFilter(int filterID, std::string key, GPUDB_Value & value, GPUDB_Type & type, GPUDB_COMP comp) {
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
int GPUDB::advanceFilter(int filterID) {
    return filters.advanceFilter(filterID);
}

// Errors: -6
int GPUDB::deleteFilter(int filterID) {
    return filters.removeFilter(filterID);
}

// ********************************************************************************
// Querying

// Errors:
std::vector<GPUDB_QueryResult> GPUDB::query(int filterID) {
    std::vector<Doc> resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID)).front();
    return translateDoc(resultDoc);
}

GPUDB_QueryResult GPUDB::translateDoc(Doc resultDoc) {
    // Set up the parent doc
    GPUDB_QueryResult userDoc;

    ResultKV newKV;
    newKV.key = StringConversion::intToString(resultDoc.kvPair.key);
    newKV.type = resultDoc.kvPair.valType;
    newKV.value = dataToValue(resultDoc.kvPair.data, newKV.type);
    userDoc.kv = &newKV;

    // Handle the children
    if (!resultDoc.children.empty()) {
        for (std::list<Doc>::iterator it = resultDoc.children.begin(); it != resultDoc.children.end(); it++) {
            GPUDB_QueryResult childResult = translateDoc(*it);
        }
    }

    return userDoc;
}

GPUDB_Value GPUDB::dataToValue(GPUDB_Data data, GPUDB_Type type) {
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
int GPUDB::updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type & type) {
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

void GPUDB::deleteFromDoc(int filterID) {
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

void GPUDB::flattenDoc(Doc d, std::list<Entry> * targetEntryList) {
    targetEntryList->push_back(d.kvPair);
    if (!d.children.empty()) {
        for (std::list<Doc>::iterator it = d.children.begin(); it != d.children.end(); it++) {
            flattenDoc(*it, targetEntryList);
        }
    }
    return;
}

// ********************************************************************************
// Testing

int main() {
    printf("Running main in top level API\n");
    return 0;
}

