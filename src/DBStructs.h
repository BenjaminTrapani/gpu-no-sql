//
// GPUDB Structures
//

#ifndef SRC_DBSTRUCTS_H
#define SRC_DBSTRUCTS_H

#include "presets.h"
#include <vector>

// Data Related Structures

enum GPUDB_Type {
    GPUDB_BLN,
    GPUDB_INT,
    GPUDB_FLT,
    GPUDB_CHAR,
    GPUDB_STR,
    GPUDB_BGV,
    GPUDB_DOC,
    GPUDB_ANY
};

union GPUDB_Data {
    bool b;
    int n;
    float f;
    char c;
    long long int s;
    long long int bigVal;
    int d; // Will be -1
};

// Entry Related Structures

typedef struct Entry {
    unsigned long long int id;
    long long int key;
    GPUDB_Type valType;
    GPUDB_Data data;
    unsigned long long int parentID;

    // Zero's memory
    Entry():id(0), key(0), valType(GPUDB_INT), parentID(0){
        data.bigVal = 0;
    }

    bool operator<(const Entry & val)const {
        return true; // TODO smarter comparison
    }

    bool operator==(const Entry & val)const {
        return key == val.key && valType == val.valType && data.bigVal == val.data.bigVal;
    }

} GPUDB_Entry;

typedef struct QueryResult {
    char key[16];
    GPUDB_Type valType;
    GPUDB_Data data;
    std::vector<struct QueryResult> children;
} GPUDB_QueryResult;

// Element Related Structures

#endif // SRC_DBSTRUCTS_H
