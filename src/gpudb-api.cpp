//
// User Level API Implementation
//

#include "gpudb-api.hpp"
#include "stdio.h"
#include <cstdio>
#include "presets.hpp"

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
int GPU_NOSQL_DB::getDoc(std::vector<std::string> strings) {
    return docs.addDoc(strings);
}

int GPU_NOSQL_DB::deleteDocRef(int docID) {
    return docs.removeDoc(docID);
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
    return newDocID;
}

int GPU_NOSQL_DB::addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type type) {
    int res = addtoDocNoSync(docID, key, value, type);
    driver.syncCreates();
    retun res;
}

int GPU_NOSQL_DB::addToDocNoSync(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type type) {
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
    int success = setEntryVal(&newEntry, value, type);
    if (success != 0) {
        return -1; // TODO error code
    }

    driver.create(newEntry);

    return 0;
}

int GPU_NOSQL_DB::setEntryVal(Entry *entry, GPUDB_Value & value, GPUDB_Type type) {
    entry.data.bigVal = 0;
    if (type == GPUDB_BLN) {
        entry->data->b = value.b;
    } else if (type == GPUDB_INT) {
        entry->data->n = value.n;
    } else if (type == GPUDB_FLT) {
        entry->data->f = value.f;
    } else if (type == GPUDB_CHAR) {
        entry->data->c = value.c;
    } else if (type == GPUDB_STR) {
        entry->data->s = value.s;
    } else if (type == GPUDB_BGV) {
        entry->data->bigVal = value.bigVal;
    } else {
        return -1;
    }
    return 0;
}

int GPU_NOSQL_DB::batchAdd(int docID, std::vector<std::string> & keys, std::vector<std::string> & values,
                           std::vector<GPUDB_Type> types) {

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

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> key) {
    // TODO
    // Translate key into an Entry for a search
    Entry newEntry;
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry);
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> key, GPUDB_Value & value, GPUDB_COMP comp) {
    // TODO
    // Translate key and value into an Entry for a search
    Entry newEntry;
    // Add new entry to given filter
    return filters.addToFilter(filterID, newEntry, comp); // TODO properly use comp here
}

int GPU_NOSQL_DB::advanceFilter(int filterID) {
    return filters.advanceFilter(filterID);
}

int GPU_NOSQL_DB::deleteFilter(int filterID) {
    return filters.removeFilter(filterID);
}

// ********************************************************************************
// Querying

GPUDB_QueryResult GPU_NOSQL_DB::query(int filterID) {
    // Get the resulting document
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(filterID));
    // TODO populate the user format with the result doc
    GPUDB_QueryResult r;
    return r;
}

// ********************************************************************************
// Updating
// must give a filter that does not hit a document
int GPU_NOSQL_DB::updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type type) {
    // Get Matching Entry
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(fitlerID));

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
    int dataSetRes = setEntryVal(&revisedEntry, value, type);
    if (dataSetRes != 0) {
        return -1; // TODO error code bad value
    }

    // Update the entry
    driver.update(oldEntry, revisedEntry);

    return 0; // TODO error code for success
}

std::vector<Doc> GPU_NOSQL_DB::flattenDoc(Doc d, bool start) {
    std::vector<Doc> resultDoc;
    return resultDoc; // TODO
}

// ********************************************************************************
// Deleting

int GPU_NOSQL_DB::deleteFromDoc(int filterID) {
    // Get Matching Docs
    Doc resultDoc = driver.getDocumentsForFilterSet(filters.getFilter(fitlerID));
    // flatten doc into single vector
    std::vector<Entry> allEntries = flattenDoc(resultDoc, true);

    // TODO need driver changes
    // implementation - delete a list of entries, not just a single entry
    // goal: batch deletes, much more efficient
    // driver.deleteAll(allEntries);
    for (std::vector<Entry>::iterator it = allEntries.begin(); it != allEntries.end(); it++) {
        driver.deleteAll(*it);
    }

    return -1; // TODO error codes for entire function
}

// ********************************************************************************
// Testing

int main() {
    printf("Running main in top level API\n");
    return 0;
}

