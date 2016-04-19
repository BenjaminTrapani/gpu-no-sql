//
// User Level API Implementation
//

#include "gpudb-api.h"
#include "stdio.h"
#include <cstdio>
// TODO check formatting on this, what is needed
#include "gpudbdriver.h"

using namespace GPUDB;

// ********************************************************************************
// Document Identification

int GPU_NOSQL_DB::getRootDoc() {
    return -1;
}

int GPU_NOSQL_DB::getDoc(std::vector<std::string> strings) {
    return -1;
}

// ********************************************************************************
// Creation

int GPU_NOSQL_DB::newDoc(int docID, std::string key) {
    return -1;
}

int GPU_NOSQL_DB::addToDoc(int docID, std::string & key, std::string & value, GPUDB_Type type) {
    return -1;
}

int GPU_NOSQL_DB::batchAdd(int docID, std::vector<std::string> & keys, std::vector<std::string> & values, GPUDB_Type type) {
    return -1;
}

int GPU_NOSQL_DB::commitDocTree(int docID) {
    return -1;
}

// ********************************************************************************
// Filter Creation

int GPU_NOSQL_DB::newFilter(int docID) {
    return -1;
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> keys) {
    return -1;
}

int GPU_NOSQL_DB::addToFilter(int filterID, std::vector<std::string> keys, std::string & value, GPUDB_COMP comp) {
    return -1;
}

// ********************************************************************************
// Querying

GPUDB_QueryResult GPU_NOSQL_DB::query(int filterID) {
    GPUDB_QueryResult r;
    return r;
}

// ********************************************************************************
// Updating - single filter only

int GPU_NOSQL_DB::updateOnDoc(int filterID, std::string & value) {
    return -1;
}

int GPU_NOSQL_DB::updateOnDoc(int filterID, std::string & value, GPUDB_Type type) {
    return -1;
}

// ********************************************************************************
// Deleting

int GPU_NOSQL_DB::deleteFromDoc(int filterID) {
    return -1;
}

int main() {
    printf("Running main in top level API\n");
    return 0;
}

