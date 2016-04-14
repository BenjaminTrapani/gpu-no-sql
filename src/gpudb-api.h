//
// Headers for the user level API
//

#ifndef GPU_NO_SQL_GPUDB_API_H
#define GPU_NO_SQL_GPUDB_API_H

#include <string>
#include "Entry.h"
#include "QueryResult.h"
#include "CoparatorType.h
"

// Document Identification
int getRootDoc();
int getDoc(std::vector<std::string> strings);

// Creation
int newDoc(int docID, std::string key);
int addToDoc(int docID, std::string & key, std::string & value, GPUDB_Type type);
int sendAdditions(int docID);

// Reading

// TODO

// Top Down Searching
// 1. Find Doc ID
// 2. Filter By Entries with that Parent
int newFilter(int docID);
int addToFilter(int filterID, std::vector<std::string> keys, std::string & value, GPUDB_COMP comp);
QueryResult query(int filterID);

// Updating
int updateOnDoc(int docID, std::string & key, std::string & value);
int updateOnDoc(int docID, std::string & key, std::string & value, GPUDB_Type type);

// Deleting
int deleteFromDoc(int docID, std::string & key);


#endif //GPU_NO_SQL_GPUDB_API_H
