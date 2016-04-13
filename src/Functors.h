//
// Created by Benjamin Trapani on 4/13/16.
//

#ifndef GPU_NO_SQL_FUNCTORS_H
#define GPU_NO_SQL_FUNCTORS_H

namespace GPUDB{
    struct IsPartialTupleMatch : thrust::unary_function<CoreTupleType,bool>{
        inline IsPartialTupleMatch(const CoreTupleType & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const CoreTupleType & val)const{
            return val == _filter;
        }

    private:
        const CoreTupleType _filter;
    };

    struct IsFullTupleMatch : thrust::unary_function<CoreTupleType, bool>{
        inline IsFullTupleMatch(const CoreTupleType & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const CoreTupleType & val)const{
            return val.fullCompare(_filter);
        }

    private:
        CoreTupleType _filter;
    };

    struct ExtractParentID : thrust::unary_function<CoreTupleType, GPUSizeType>{
        __device__ __host__
        inline GPUSizeType operator() (const CoreTupleType & val)const{
            return val.parentID;
        }
    };

    struct ModifyTuple : thrust::unary_function<CoreTupleType, CoreTupleType>{
        inline ModifyTuple(const CoreTupleType & updates):_updates(updates){}

        __device__ __host__
        inline CoreTupleType operator() (const CoreTupleType & val)const{
            CoreTupleType result = val;
            result.data.bigVal = _updates.data.bigVal;
            result.valType = _updates.valType;
            return result;
        }
    private:
        const CoreTupleType _updates;
    };

    struct FetchTupleWithParentIDs : thrust::unary_function<CoreTupleType,bool>{
        inline FetchTupleWithParentIDs(CoreTupleType* validIndices,
                                       const size_t indexToExamine, const CoreTupleType & filter):
                _validIndices(validIndices), _indexToExamine(indexToExamine), _filter(filter){
        }

        __device__ __host__
        inline bool operator()(const CoreTupleType & ival)const{
            return _validIndices[_indexToExamine].parentID == ival.id && ival == _filter;
        }

    private:
        CoreTupleType * _validIndices;
        const size_t _indexToExamine;
        const CoreTupleType _filter;
    };

    struct FetchDescendentTuple : thrust::unary_function<CoreTupleType, bool>{
        inline FetchDescendentTuple(const CoreTupleType * desiredParentID): _desiredParentID(desiredParentID){}

        __device__ __host__
        inline bool operator()(const CoreTupleType & ival)const{
            return ival.parentID == _desiredParentID->id && ival.parentID!=0;
        }

    private:
        const CoreTupleType * _desiredParentID;
    };
}

#endif //GPU_NO_SQL_FUNCTORS_H
