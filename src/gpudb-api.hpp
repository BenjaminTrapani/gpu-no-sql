//
// Headers for the user level API
//

#ifndef GPU_NO_SQL_GPUDB_API_H
#define GPU_NO_SQL_GPUDB_API_H

#include <string>
#include <vector>
#include <list>
#include "Entry.hpp"
#include "QueryResult.hpp"
#include "ComparatorType.hpp"
#include "gpudbdriver.hpp"
#include "DocMap.hpp"
#include "FilterMap.hpp"
#include "presets.hpp"

using namespace GPUDB;

class GPUDB {
public:
    GPUDB();

    // Document Identification
    int getRootDoc();
    int getDoc(std::vector<std::string> & strings);
    int deleteDocRef(int docID);

    // Creation
    // Returns the new Doc ID
    int newDoc(int docID, std::string & key);
    // Returns an error/success code
    int addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type);
    // Returns an error/success code
    long int batchAdd(int docID, std::vector<std::string> & keys, std::vector<GPUDB_Value> & values, std::vector<GPUDB_Type> & types);

    // Filter Creation and Editing

    // Returns the new filter ID on the given Doc
    int newFilter(int docID);
    // Returns an error/success code
    int addToFilter(int filterID, std::string key);
    // Returns an error/success code
    int addToFilter(int filterID, std::string key, GPUDB_Value & value, GPUDB_Type & type, GPUDB_COMP comp);
    // Move the filter to the next level
    int advanceFilter(int filterID);
    // Delete the current filter
    int deleteFilter(int filterID);


    // Querying
    std::vector<GPUDB_QueryResult> query(int filterID);

    // Updating
    // Returns an error/success code
    int updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type & type);

    // Deleting
    // Returns an error/success code
    int deleteFromDoc(int filterID);

private:
    GPUDBDriver driver;
    DocMap docs;
    FilterMap filters;

    unsigned long long int curID;

    int addToDocNoSync(unsigned long long int realDocID, std::string & key, GPUDB_Value & value, GPUDB_Type & type);

    void setEntryVal(Entry & entry, GPUDB_Value & value, GPUDB_Type & type);
    GPUDB_Value dataToValue(GPUDB_Data data, GPUDB_Type type);

    GPUDB_QueryResult translateDoc(Doc resultDoc);
    void flattenDoc(Doc d, std:: list<Entry> * targetEntryList);
};

#endif //GPU_NO_SQL_GPUDB_API_H
