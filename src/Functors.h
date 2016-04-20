//
// Comparators and Query Builders
//

#ifndef GPU_NO_SQL_FUNCTORS_H
#define GPU_NO_SQL_FUNCTORS_H

namespace GPUDB {
    struct IsPartialEntryMatch : thrust::unary_function<Entry, bool> {
        inline IsPartialEntryMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val == _filter;
        }

    private:
        const Entry _filter;
    };

    struct IsFullEntryMatch : thrust::unary_function<Entry, bool> {
        inline IsFullEntryMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val.fullCompare(_filter);
        }

    private:
        Entry _filter;
    };

    struct IsSelectedWithParentID : thrust::unary_function<Entry, bool> {
        inline IsSelectedWithParentID(const unsigned long long int desiredParentID):
                _desiredParentID(desiredParentID){}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val.selected && val.parentID == _desiredParentID;
        }

    private:
        unsigned long long int _desiredParentID;
    };

    struct FetchEntryWithParentID : thrust::unary_function<Entry,bool> {
        inline FetchEntryWithParentID(const Entry & filter, const Entry* validIndex):_filter(filter),
                                        _validIndex(validIndex) {
        }

        __device__ __host__
        inline bool operator()(const Entry & ival) const {
            return _validIndex->parentID == ival.id && ival == _filter;
        }

    private:
        const Entry _filter;
        const Entry * _validIndex;
    };

    struct FetchEntryWithChildID : thrust::unary_function<Entry,bool> {
        inline FetchEntryWithChildID(const Entry & filter, const Entry* validIndex):_filter(filter),
                                                                                     _validIndex(validIndex) {
        }

        __device__ __host__
        inline bool operator()(const Entry & ival) const {
            return _validIndex->id == ival.parentID && ival == _filter;
        }

    private:
        const Entry _filter;
        const Entry * _validIndex;
    };

    struct GetElementWithChild : thrust::unary_function<Entry,bool> {
        inline GetElementWithChild(const Entry* validIndex): _validIndex(validIndex) {}

        __device__ __host__
        inline bool operator()(const Entry & ival) const {
            return _validIndex->parentID == ival.id;
        }

    private:
        const Entry * _validIndex;
    };

    struct IsLayerNotEqualTo : thrust::unary_function<Entry, bool> {
        inline IsLayerNotEqualTo(const unsigned long int layer):_layer(layer){}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val.layer != _layer;
        }

    private:
        const unsigned long int _layer;
    };

    struct IsEntrySelected : thrust::unary_function<Entry, bool> {
        inline IsEntrySelected(const unsigned long int layer):_layer(layer) {}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val.selected && val.layer == _layer;
        }

    private:
        unsigned long int _layer;
    };

    struct IsEntrySelectedAndPartialMatched : thrust::unary_function<Entry, bool> {
        inline IsEntrySelectedAndPartialMatched(const Entry & expected):_expected(expected){}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val.selected && val == _expected;
        }

    private:
        const Entry _expected;
    };

    struct EntryLessThan : thrust::unary_function<Entry, bool> {
        inline EntryLessThan(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val) const {
            return val.key == _filter.key && val.valType == _filter.valType && val.data.bigVal < _filter.data.bigVal;
        }

        private:
            Entry _filter;
    };

    struct ExtractParentID : thrust::unary_function<Entry, GPUSizeType>{
        __device__ __host__
        inline GPUSizeType operator() (const Entry & val) const {
            return val.parentID;
        }
    };

    struct ModifyEntry : thrust::unary_function<Entry, Entry>{
        inline ModifyEntry(const Entry & updates):_updates(updates){}

        __device__ __host__
        inline Entry operator() (const Entry & val)const{
            Entry result = val;
            result.data.bigVal = _updates.data.bigVal;
            result.valType = _updates.valType;
            return result;
        }
    private:
        const Entry _updates;
    };

    struct SelectEntry : thrust::unary_function<Entry, Entry> {
        inline SelectEntry(unsigned long int layer, const bool isResultMember):_layer(layer),
                                                                               _isResultMember(isResultMember){}

        __device__ __host__
        inline Entry operator() (const Entry & val) const {
            Entry result = val;
            result.selected = true;
            result.layer = _layer;
            result.isResultMember = _isResultMember;
            return result;
        }

    private:
        const unsigned long int _layer;
        const bool _isResultMember;
    };

    struct UnselectEntry : thrust::unary_function<Entry, Entry> {
        __device__ __host__
        inline Entry operator() (const Entry & val) const {
            Entry result = val;
            result.selected = false;
            result.layer = 0;
            result.isResultMember = false;
            return result;
        }
    };

    struct FetchEntryWithParentIDs : thrust::unary_function<Entry, bool> {
        inline FetchEntryWithParentIDs(Entry* validIndices, const size_t indexToExamine, const Entry & filter):
                _validIndices(validIndices), _indexToExamine(indexToExamine), _filter(filter) {
        }

        __device__ __host__
        inline bool operator()(const Entry & ival) const {
            return _validIndices[_indexToExamine].parentID == ival.id && ival == _filter;
        }

    private:
        Entry * _validIndices;
        const size_t _indexToExamine;
        const Entry _filter;
    };

    struct FetchDescendentEntry : thrust::unary_function<Entry, bool> {
        inline FetchDescendentEntry(const Entry * desiredParentID): _desiredParentID(desiredParentID){}

        __device__ __host__
        inline bool operator()(const Entry & ival) const {
            return ival.parentID == _desiredParentID->id && ival.parentID!=0;
        }

    private:
        const Entry * _desiredParentID;
    };
}

#endif //GPU_NO_SQL_FUNCTORS_H
