//
// User Level API Implementation
//

#include "gpudb-api.hpp"
#include "stdio.h"
#include <cstdio>
#include "presets.hpp"
#include "StringConversion.hpp"

using namespace GPUDB;

GPU_NOSQL_DB::GPU_NOSQL_DB() {
    curID = 0;
    GPUDBDriver d();
    driver = d;
    DocMap m(&driver);
    docs = m;
    FilterMap fm();
    filters = fm;
}

// ********************************************************************************
// Document Identification

int GPU_NOSQL_DB::getRootDoc() {
    return 0;
}

// Returns -1 when not enough space, otherwise returns the docID
int GPU_NOSQL_DB::getDoc(std::vector<std::string> & strings) {
    return docs.addDoc(strings);
}

int GPU_NOSQL_DB::deleteDocRef(int docID) {
    return docs.removeDoc(docID);
}

// ********************************************************************************
// Creation

// Creates a document inside the given doc from ID with key "key"
int GPU_NOSQL_DB::newDoc(int docID, std::string & key) {
    // Create the new path
    std::vector<std::string> newPath = docs.getPath(docID);
    newPath.push_back(key);

    // Create the actual new doc
    unsigned long long int parentID = docs.getDoc(docID);
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.valType = GPUDB_DOC;
    if (key.length() < MAX_STRING_SIZE) {
        newEntry.key = stringToInt(key.c_str());
    } else {
        return -1; // TODO error code
    }

    // Add New Entry to database
    driver.create(newEntry);
    driver.syncCreates();

    // Get a docID for the path
    int newDocID = getDoc(newPath);

    // return the new doc ID
    return newDocID;
}

int GPU_NOSQL_DB::addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {
    int res = addToDocNoSync(docID, key, value, type); // TODO handle error code
    driver.syncCreates();
    return res;
}

int GPU_NOSQL_DB::addToDocNoSync(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type) {
    // Create a new Entry
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.parentID = docs.getDoc(docID);
    if (key.length() < MAX_STRING_SIZE) {
        newEntry.key = stringToInt(key.c_str());
    } else {
        return -1; // TODO error code
    }
    newEntry.valType = type;
    setEntryVal(newEntry, value, type);

    driver.create(newEntry);

    return 0;
}

void GPU_NOSQL_DB::setEntryVal(Entry & entry, GPUDB_Value & value, GPUDB_Type & type) {
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
        entry.data.s = value.s;
    } else {
        entry.data.bigVal = value.bigVal;
    }
}

int GPU_NOSQL_DB::batchAdd(int docID, std::vector<std::string> & keys, std::vector<GPUDB_Value> & values, std::vector<GPUDB_Type> & types) {

    int keySize = keys.size();
    if (keySize != values.size() && keySize != types.size()) {
        return -1; // TODO error code
    }
    for (int i = 0; i < keySize; i += 1) {
        int res = addToDocNoSync(docID, keys.at(i), values.at(i), types.at(i));
        if (res != 0) {
            return -i; // returns the negative of the value it fialed on - all other values are not processed
        }
    }
    driver.syncCreates();
    return 0;
}

// ********************************************************************************
// Filter Creation

int GPU_NOSQL_DB::newFilter(int docID) {
    return filters.newFilter(docs.getFilterSet(docID));
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::string key) {
    // Translate key into an Entry for a search
    Entry newEntry;
    newEntry.key = stringToInt(key.c_str());
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, KEY_ONLY);
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::string key, GPUDB_Value & value, GPUDB_Type & type, GPUDB_COMP comp) {
    // Translate key and value into an Entry for a search
    Entry newEntry;
    newEntry.key = stringToInt(key.c_str());
    setEntryVal(newEntry, value, type);
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, comp);
}

int GPU_NOSQL_DB::advanceFilter(int filterID) {
    return filters.advanceFilter(filterID);
}

int GPU_NOSQL_DB::deleteFilter(int filterID) {
    return filters.removeFilter(filterID);
}

// ********************************************************************************
// Querying

std::vector<GPUDB_QueryResult> GPU_NOSQL_DB::query(int filterID) {
    // Get the resulting document
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID));

    std::vector<GPUDB_QueryResult> allResults;

    for (std::vector<Doc>::iterator it = resultDoc.begin(); it != resultDoc.end(); it++) {
        GPUDB_QueryResult result = translateDoc(*it);
        allResults.push_back(result);
    }


    return allResults;
}

GPUDB_QueryResult GPU_NOSQL_DB::translateDoc(Doc resultDoc) {
    // Set up the parent doc
    GPUDB_QueryResult userDoc;

    ResultKV newKV;
    newKV.key = intToString(resultDoc.kvPair.key);
    newKV.type = resultDoc.kvPair.valType;
    newKV.value = dataToValue(resultDoc.kvPair.data, newKV.type);
    userDoc.kv = newKV;

    // Handle the children
    if (!resultDoc.children.empty()) {
        for (std::list<Doc>::iterator it = resultDoc.children.begin(); it != resultDoc.children.end(); it++) {
            GPUDB_QueryResult childResult = translateDoc(*it);
        }
    }

    return userDoc;
}

GPUDB_Value GPU_NOSQL_DB::dataToValue(GPUDB_Data data, GPUDB_Type type) {
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
        v.s = data.s;
    } else {
        v.bigVal = data.bigVal;
    }
    return v;
}

// ********************************************************************************
// Updating
// must give a filter that does not hit a document
int GPU_NOSQL_DB::updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type & type) {
    // Get Matching Entry
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID)).front();

    // Check that it is not a doc
    Entry oldEntry = resultDoc.kvPair;
    if (oldEntry.valType == GPUDB_DOC) {
        return -1; // TODO error code does not hit a document
    }

    // Create the revised entry
    Entry revisedEntry;
    revisedEntry.parentID = oldEntry.parentID;
    revisedEntry.id = oldEntry.id;
    revisedEntry.key = oldEntry.key;
    revisedEntry.valType = type;
    setEntryVal(revisedEntry, value, type);

    // Update the entry
    driver.update(oldEntry, revisedEntry);

    return 0; // TODO error code for success
}

// ********************************************************************************
// Deleting

int GPU_NOSQL_DB::deleteFromDoc(int filterID) {
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

    return -1; // TODO error codes for entire function
}

void GPU_NOSQL_DB::flattenDoc(Doc d, std::list<Entry> * targetEntryList) {
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

