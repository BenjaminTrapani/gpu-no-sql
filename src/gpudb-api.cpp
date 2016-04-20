//
// User Level API Implementation
//

#include "gpudb-api.h"
#include "stdio.h"
#include <cstdio>
#include "gpudbdriver.h"

using namespace GPUDB;

GPU_NOSQL_DB::GPU_NOSQL_DB() {
    curID = 0;
}

// ********************************************************************************
// Document Identification

int GPU_NOSQL_DB::getRootDoc() {
    return 0;
}

// Returns -1 when not enough space, otherwise returns the docID
int GPU_NOSQL_DB::getDoc(std::vector<std::string> strings) {
    return docs.addDoc(strings);
}

void GPU_NOSQL_DB::deleteDocRef(int docID) {
    docs.removeDoc(docID);
}

// ********************************************************************************
// Creation

// Creates a document inside the given doc from ID with key "key"
int GPU_NOSQL_DB::newDoc(int docID, std::string key) {
    // Create the new path
    std::vector<std::string> newPath = docs.getPath(docID);
    newPath.push_back(key);

    // Create the actual new doc
    unsigned long long int parentID = docs.getDoc(docID);
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.valType = GPUDB_DOC;
    if (key.length() < 16) {
        strcpy(newEntry.key, key.c_str());
    } else {
        return -1; // TODO error code
    }

    // Add New Entry to database
    driver.create(newEntry);
    driver.syncCreates();

    // Get a docID for the path
    int newDocID = getDoc(newPath);

    // return the new doc ID
    return newDocID; // TODO
}

int GPU_NOSQL_DB::addToDoc(int docID, std::string & key, std::string & value, GPUDB_Type type) {
    int res = addtoDocNoSync(docID, key, value, type);
    driver.syncCreates();
    retun res;
}

int GPU_NOSQL_DB::addToDocNoSync(int docID, std::string & key, std::string & value, GPUDB_Type type) {
    // Create a new Entry
    Entry newEntry;
    newEntry.id = curID;
    curID += 1;
    newEntry.parentID = docs.getDoc(docID);
    if (key.length() < 16) {
        strcpy(newEntry.key, key.c_str());
    } else {
        return -1; // TODO error code
    }
    newEntry.valType = type;
    setEntryVal(newEntry, value, type);

    driver.create(newEntry);

    return 0;
}

void GPU_NOSQL_DB::setEntryVal(Entry & entry, std::string &value, GPUDB_Type type) {
    if (type == GPUDB_BLN) {
        entry.data.b = boost::lexical_cast<bool>("1");
    } else if (type == GPUDB_INT) {
        entry.data.n = std::stoi(value);
    } else if (type == GPUDB_FLT) {
        entry.data.f = std::ftoi(value);
    } else if (type == GPUDB_CHAR) {
        entry.data
    } else if (type == GPUDB_STR) {

    } else if (type == GPUDB_BGV) {

    } else {
        // this was an error
        return;
    }
}

int GPU_NOSQL_DB::batchAdd(int docID, std::vector<std::string> & keys, std::vector<std::string> & values, GPUDB_Type type) {
    return -1; // TODO
}

// ********************************************************************************
// Filter Creation

int GPU_NOSQL_DB::newFilter(int docID) {
    return -1; // TODO
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> keys) {
    return -1; // TODO
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> keys, std::string & value, GPUDB_COMP comp) {
    return -1; // TODO
}

// ********************************************************************************
// Querying

GPUDB_QueryResult GPU_NOSQL_DB::query(int filterID) {
    GPUDB_QueryResult r;
    return r; // TODO
}

// ********************************************************************************
// Updating - single filter only

int GPU_NOSQL_DB::updateOnDoc(int filterID, std::string & value) {
    return -1; // TODO
}

int GPU_NOSQL_DB::updateOnDoc(int filterID, std::string & value, GPUDB_Type type) {
    return -1; // TODO
}

// ********************************************************************************
// Deleting

int GPU_NOSQL_DB::deleteFromDoc(int filterID) {
    return -1; // TODO
}

// ********************************************************************************
// Testing

// TODO
int main() {
    printf("Running main in top level API\n");
    return 0;
}

