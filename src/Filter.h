//
// Created by Benjamin Trapani on 4/22/16.
//

#ifndef GPU_NO_SQL_FILTER_H
#define GPU_NO_SQL_FILTER_H

#include "Entry.hpp"
#include "ComparatorType.hpp"

namespace GPUDB{
    class Filter{
    public:
        Filter():entry(), comparator(EQ){}
        Filter(const Entry & ientry, const GPUDB_COMP & comp):entry(ientry),comparator(comp){}
        Entry entry;
        GPUDB_COMP comparator;
    };
}
#endif //GPU_NO_SQL_FILTER_H
