//
// Headers for the user level API
//

// TODO
// Comparators

#ifndef GPU_NO_SQL_GPUDB_API_H
#define GPU_NO_SQL_GPUDB_API_H

#include <string>
#include <vector>
#include "Entry.h"
#include "QueryResult.h"
#include "ComparatorType.h"

using namespace GPUDB;

class GPU_NOSQL_DB {
public:
    // Document Identification
    int getRootDoc();
    int getDoc(std::vector<std::string> strings);


    // Creation
    // Returns the new Doc ID
    int newDoc(int docID, std::string key);
    // Returns an error/success code
    int addToDoc(int docID, std::string & key, std::string & value, GPUDB_Type type);
    // Returns an error/success code
    int batchAdd(int docID, std::vector<std::string> & keys, std::vector<std::string> & values, GPUDB_Type type);
    // Returns an error/success code
    int commitDocTree(int docID);

    // Filter Creation
    // Returns the new filter ID on the given Doc
    int newFilter(int docID);
    // Returns an error/success code
    int addToFilter(int filterID, std::vector<std::string> keys);
    // Returns an error/success code
    int addToFilter(int filterID, std::vector<std::string> keys, std::string & value, GPUDB_COMP comp);

    // Querying
    GPUDB_QueryResult query(int filterID);

    // Updating - single filter only or error out
    // Returns an error/success code
    int updateOnDoc(int filterID, std::string & value);
    // Returns an error/success code
    int updateOnDoc(int filterID, std::string & value, GPUDB_Type type);

    // Deleting
    // Returns an error/success code
    int deleteFromDoc(int filterID);
private:

};



#endif //GPU_NO_SQL_GPUDB_API_H
