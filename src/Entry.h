//
// Created by Benjamin Trapani on 4/12/16.
//

#ifndef GPU_NO_SQL_ENTRY_H
#define GPU_NO_SQL_ENTRY_H

namespace GPUDB {
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
        int d;
    };

// Entry Related Structures

    typedef struct Entry {
        unsigned long long int id;
        long long int key;
        GPUDB_Type valType;
        GPUDB_Data data;
        unsigned long long int parentID;
        //runtime data used in searches
        bool selected;
        unsigned long int layer;

        // Zero's memory
        Entry() : id(0), key(0), valType(GPUDB_INT), parentID(0), selected(false), layer(0) {
            data.bigVal = 0;
        }

        bool operator<(const Entry &val) const {
            return true; // TODO smarter comparison
        }

        inline bool operator==(const Entry &val) const {
            return key == val.key && valType == val.valType && data.bigVal == val.data.bigVal;
        }

        inline bool fullCompare(const Entry &other) const {
            return other.id == id && other.key == key && other.valType == valType && other.data.bigVal == data.bigVal &&
                   other.parentID == parentID;
        }


    } GPUDB_Entry;

}
#endif //GPU_NO_SQL_ENTRY_H
