//
// Entry Data Definition
//

#ifndef GPU_NO_SQL_ENTRY_H
#define GPU_NO_SQL_ENTRY_H

#include "TypesValues.hpp"
#include "presets.hpp"

namespace GPUDB {

    typedef struct Entry {

        unsigned long long int id;
        unsigned long long int parentID;
        long long int key[STRING_SIZE_INT];

        GPUDB_Type valType;
        GPUDB_Data data;

        // runtime data used in searches
        bool selected;
        unsigned long int layer;
        bool isResultMember;

        // Zeros memory
        Entry() : id(0), key({0}), valType(GPUDB_INT), parentID(0), selected(false), layer(0),
                  isResultMember(false) {
            data.bigVal = 0;
        }

        inline bool operator<(const Entry &val) const {
            return key[0] == val.key[0] && key[1] == val.key[1] &&
                    valType == val.valType && data.bigVal < val.data.bigVal;
        }

        inline bool operator<=(const Entry &val) const {
            return key[0] == val.key[0] && key[1] == val.key[1] &&
                   valType == val.valType && data.bigVal <= val.data.bigVal;
        }

        inline bool operator==(const Entry &val) const {
            return key[0] == val.key[0] && key[1] == val.key[1] && valType == val.valType &&
                    data.bigVal == val.data.bigVal;
        }

        inline bool operator>=(const Entry &val) const {
            return key[0] == val.key[0] && key[1] == val.key[1] &&
                   valType == val.valType && data.bigVal >= val.data.bigVal;
        }

        inline bool operator>(const Entry &val) const {
            return key[0] == val.key[0] && key[1] == val.key[1] &&
                   valType == val.valType && data.bigVal > val.data.bigVal;
        }

        inline bool fullCompare(const Entry &other) const {
            return other.id == id && other.key == key && other.valType == valType && other.data.bigVal == data.bigVal &&
                   other.parentID == parentID;
        }

    } GPUDB_Entry;
}

#endif //GPU_NO_SQL_ENTRY_H
