//
// Created by Benjamin Trapani on 4/13/16.
//

#ifndef GPU_NO_SQL_FILTERSET_HPP
#define GPU_NO_SQL_FILTERSET_HPP
#include <vector>
#include "Entry.h"

namespace GPUDB {
    typedef std::vector<Entry> FilterGroup;
    typedef std::vector<FilterGroup> FilterSet;
}

#endif //GPU_NO_SQL_FILTERSET_HPP
