//
// Utilities for Entries
//

#ifndef GPU_NO_SQL_ENTRYUTILS_H
#define GPU_NO_SQL_ENTRYUTILS_H

#include <algorithm>
#include <string.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include "Entry.hpp"

namespace GPUDB {
    class EntryUtils {
    public:
        template<class T>
        static void assignKeyToEntry(Entry &entry, const T &val) {
            union TempUnion{
                T valToFill;
                long long int resultBuf[Entry::KeyLen];
            };
            TempUnion aUnion;
            memset(aUnion.resultBuf, 0, sizeof(aUnion.resultBuf));
            aUnion.valToFill = val;
            memcpy(entry.key, aUnion.resultBuf, sizeof(T));
        }
    };
}
#endif //GPU_NO_SQL_ENTRYUTILS_H
