//
// Comparators and Query Builders
//

#ifndef GPU_NO_SQL_FUNCTORS_H
#define GPU_NO_SQL_FUNCTORS_H

namespace GPUDB{
    struct IsPartialEntryMatch : thrust::unary_function<Entry,bool>{
        inline IsPartialEntryMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val == _filter;
        }

    private:
        const Entry _filter;
    };

    struct IsFullEntryMatch : thrust::unary_function<Entry, bool>{
        inline IsFullEntryMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val.fullCompare(_filter);
        }

    private:
        Entry _filter;
    };

    struct ExtractParentID : thrust::unary_function<Entry, GPUSizeType>{
        __device__ __host__
        inline GPUSizeType operator() (const Entry & val)const{
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

    struct FetchEntryWithParentIDs : thrust::unary_function<Entry,bool>{
        inline FetchEntryWithParentIDs(Entry* validIndices,
                                       const size_t indexToExamine, const Entry & filter):
                _validIndices(validIndices), _indexToExamine(indexToExamine), _filter(filter){
        }

        __device__ __host__
        inline bool operator()(const Entry & ival)const{
            return _validIndices[_indexToExamine].parentID == ival.id && ival == _filter;
        }

    private:
        Entry * _validIndices;
        const size_t _indexToExamine;
        const Entry _filter;
    };

    struct FetchDescendentEntry : thrust::unary_function<Entry, bool>{
        inline FetchDescendentEntry(const Entry * desiredParentID): _desiredParentID(desiredParentID){}

        __device__ __host__
        inline bool operator()(const Entry & ival)const{
            return ival.parentID == _desiredParentID->id && ival.parentID!=0;
        }

    private:
        const Entry * _desiredParentID;
    };
}

#endif //GPU_NO_SQL_FUNCTORS_H
