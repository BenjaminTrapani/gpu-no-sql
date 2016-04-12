//
// Created by Benjamin Trapani on 4/12/16.
//

#ifndef GPU_NO_SQL_DOC_H
#define GPU_NO_SQL_DOC_H
#include <vector>
#include "Entry.h"

namespace GPUDB {
    class Doc {
    public:
        Doc(){}
        Doc(const Entry & entry):kvPair(entry){}

        Entry kvPair;
        std::vector<Doc> children;
    };
}

#endif //GPU_NO_SQL_DOC_H
