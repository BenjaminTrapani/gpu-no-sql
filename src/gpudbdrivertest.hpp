//
// Created by Benjamin Trapani on 4/19/16.
//

#ifndef GPU_NO_SQL_GPUDBDRIVERTEST_H
#define GPU_NO_SQL_GPUDBDRIVERTEST_H
#include "Doc.hpp"

class GPUDBDriverTest{
public:
    void runTests();
private:
    void runDeepNestingTests();
    void runTwoKeyTest();
    void runLargeResultTest();
};

#endif //GPU_NO_SQL_GPUDBDRIVERTEST_H
