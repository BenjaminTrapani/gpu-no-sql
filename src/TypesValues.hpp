//
// Created by Adrian Kant on 4/22/16.
//

#ifndef GPU_NO_SQL_TYPESVALUES_H
#define GPU_NO_SQL_TYPESVALUES_H

#endif //GPU_NO_SQL_TYPESVALUES_H

#include "presets.hpp"

namespace GPUDB {
    enum GPUDB_Type {
        GPUDB_BLN,
        GPUDB_INT,
        GPUDB_FLT,
        GPUDB_CHAR,
        GPUDB_STR,
        GPUDB_BGV,
        GPUDB_DOC
    };

    union GPUDB_Value {
        bool b;
        int n;
        float f;
        char c;
        char s[MAX_STRING_SIZE/2];
        long long int bigVal;
    };

    union GPUDB_Data {
        bool b;
        int n;
        float f;
        char c;
        char s[MAX_STRING_SIZE/2];
        long long int bigVal;
    };
}