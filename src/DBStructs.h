//
// GPUDB Structures
//

#ifndef SRC_DBSTRUCTS_H
#define SRC_DBSTRUCTS_H

#include "presets.h"

// Data Related Structures

enum DATA_TYPE {
    GPUDB_INT,
    GPUDB_FLT,
    GPUDB_CHAR,
    GPUDB_STR
} GPUDB_Type

union Data {
    int num;
    float f;
    char c;
    char str[MAX_STRING_SIZE];
} GPUDB_Data;

// Entry Related Structures

typedef struct Entry {
    char key[MAX_KEY_LENGTH];
    GPUDB_Type valType;
    GPUDB_Data data;
} GPUDB_Entry;

typedef struct EntryNode {
    GPUDB_Entry entry;
    struct EntryNode *next;
} GPUDB_EntryNode;

// Schema Related Structures

typedef struct Schema {
    int id;
    KeyNode *keys;
} GPUDB_Schema;

// Element Related Structures

typedef struct Element {
    GPUDB_Schema schema = -1;
    GPUDB_EntryNode *pairs;
} GPUDB_Element;

// Result Related Structures

typedef struct ElementNode {

} GPUDB_ElementNode;

typedef struct QueryResult {
    ElementNode result;
    struct QueryResult *next;
} GPUDB_QueryResult;

#endif // SRC_DBSTRUCTS_H
