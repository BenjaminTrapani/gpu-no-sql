//
// Created by Benjamin Trapani on 4/19/16.
//

#ifndef GPU_NO_SQL_GPUDBDRIVERTEST_H
#define GPU_NO_SQL_GPUDBDRIVERTEST_H
#include "Doc.h"

class GPUDBDriverTest{
public:
    void runTests();
private:
    void runDeepNestingTests();
    void generateNestedDoc(size_t nestings, GPUDB::Doc * parent, size_t beginIndex);
};

#endif //GPU_NO_SQL_GPUDBDRIVERTEST_H
