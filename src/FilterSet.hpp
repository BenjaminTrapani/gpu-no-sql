//
// Filter Namespace Info
//

#ifndef GPU_NO_SQL_FILTERSET_HPP
#define GPU_NO_SQL_FILTERSET_HPP
#include <vector>
#include "Entry.hpp"
#include "Filter.hpp"

namespace GPUDB {
    class FilterGroup {
    public:
        FilterGroup():resultMember(false){}

        typedef std::vector<Filter>::const_iterator const_iterator;

        std::vector<Filter> group;
        bool resultMember;
    };

    typedef std::vector<FilterGroup> FilterSet;
}

#endif //GPU_NO_SQL_FILTERSET_HPP
