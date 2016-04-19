//
// Entry Data Definition
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
        GPUDB_ANY, // Only compare key values
        GPUDB_VAL, // Only compare vals and val types
    };

    union GPUDB_Data {
        bool b;
        int n;
        float f;
        char c;
        long long int s;
        long long int bigVal;
    };

// Entry Related Structures

    typedef struct Entry {

        unsigned long long int id;
        unsigned long long int parentID;

        long long int key;

        GPUDB_Type valType;
        GPUDB_Data data;

        // runtime data used in searches
        bool selected;
        unsigned long int layer;
        bool isResultMember;

        // Zero's memory
        Entry() : id(0), key(0), valType(GPUDB_INT), parentID(0), selected(false), layer(0),
                  isResultMember(false) {
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
