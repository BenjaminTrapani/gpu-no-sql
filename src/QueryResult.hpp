//
// Query Result Structure
//

#ifndef GPU_NO_SQL_QUERYRESULT_H
#define GPU_NO_SQL_QUERYRESULT_H

#include "presets.hpp"
#include <list>
#include <vector>
#include "Entry.hpp"

namespace GPUDB {
    typedef struct ResultKV {
        char key[MAX_STRING_SIZE];
        GPUDB_Value value;
        GPUDB_Type type;
    };

    typedef struct QueryResult {
        ResultKV *kv;
        std::list<QueryResult> children;
    } GPUDB_QueryResult;
}

#endif //GPU_NO_SQL_QUERYRESULT_H
