//
// GPUDB Structures
//

#ifndef SRC_DBSTRUCTS_H
#define SRC_DBSTRUCTS_H

#include "presets.h"
#include <vector>

// Data Related Structures

enum GPUDB_Type {
    GPUDB_INT,
    GPUDB_FLT,
    GPUDB_CHAR,
    GPUDB_STR
};

union GPUDB_Data {
    int num;
    float f;
    char c;
    long long bigVal;
};

// Entry Related Structures

typedef struct Entry {
    unsigned int key;
    GPUDB_Type valType;
    GPUDB_Data data;

    Entry():key(0), valType(GPUDB_INT){
        data.bigVal = 0;
    }

    bool operator<(const Entry & val)const{
        return true; //TODO smarter comparison
    }

    bool operator==(const Entry & val)const{
        return key == val.key && data.bigVal == val.data.bigVal;
    }

} GPUDB_Entry;

typedef struct QueryResult{
    const Entry * entries;
    const unsigned int numEntries;
    QueryResult(const Entry * ientries, const unsigned int inumEntries):entries(ientries),
                                                                        numEntries(inumEntries){}
}GPUDB_QueryResult;

typedef struct Schema {
    char key[MAX_KEY_LENGTH];
    std::vector<GPUDB_Entry*> entries;
}GPUDB_Schema;

typedef struct EntryNode {
    GPUDB_Entry entry;
    struct EntryNode *next;
} GPUDB_EntryNode;

// Element Related Structures

typedef struct Element {
    int schema;
    GPUDB_EntryNode *pairs;
} GPUDB_Element;

// Result Related Structures

typedef struct ElementNode {

} GPUDB_ElementNode;

#endif // SRC_DBSTRUCTS_H
