//
// Headers for the user level API
//

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

    // Filter Creation
    int newFilter(int docID);
    int addToFilter(int filterID, std::vector<std::string> keys);
    int addToFilter(int filterID, std::vector<std::string> keys, std::string & value, GPUDB_COMP comp);

    // Creation
    int newDoc(int docID, std::string key);
    int addToDoc(int docID, std::string & key, std::string & value, GPUDB_Type type);
    int batchAdd(int docID, std::vector<std::string> & keys, std::vector<std::string> & values, GPUDB_Type type);
    bool commitDocTree(int docID);

    // Querying
    GPUDB_QueryResult query(int filterID);

    // Updating - single filter only or error out
    int updateOnDoc(int filterID, std::string & value);
    int updateOnDoc(int filterID, std::string & value, GPUDB_Type type);

    // Deleting
    int deleteFromDoc(int filterID);
private:

};



#endif //GPU_NO_SQL_GPUDB_API_H
