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
    typedef struct GPUDB_KV {
        std::string key;
        GPUDB_Value value;
        GPUDB_Type type;
    };

    typedef struct QueryResult {
        GPUDB_KV *kv;
        std::list<QueryResult> children;
    } GPUDB_QueryResult;
}

#endif //GPU_NO_SQL_QUERYRESULT_H
