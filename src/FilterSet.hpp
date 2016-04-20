//
// Filter Namespace Info
//

#ifndef GPU_NO_SQL_FILTERSET_HPP
#define GPU_NO_SQL_FILTERSET_HPP
#include <vector>
#include "Entry.h"

namespace GPUDB {
    typedef class FilterGroup {
    public:
        FilterGroup():resultMember(false){}

        typedef std::vector<Entry>::const_iterator const_iterator;

        std::vector<Entry> group;
        bool resultMember;
    };

    typedef std::vector<FilterGroup> FilterSet;
}

#endif //GPU_NO_SQL_FILTERSET_HPP
