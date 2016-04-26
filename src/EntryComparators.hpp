//
// Comparators for Entries
//

#ifndef GPU_NO_SQL_ENTRYCOMPARATORS_H
#define GPU_NO_SQL_ENTRYCOMPARATORS_H

#include "Filter.hpp"

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

    struct IsEntryGreater : thrust::unary_function<Entry, bool> {
        inline IsEntryGreater(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val > _filter;
        }
    private:
        Entry _filter;
    };

    struct IsEntryGreaterEQ : thrust::unary_function<Entry, bool> {
        inline IsEntryGreaterEQ(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val >= _filter;
        }
    private:
        Entry _filter;
    };

    struct IsEntryLess : thrust::unary_function<Entry, bool> {
        inline IsEntryLess(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val < _filter;
        }
    private:
        Entry _filter;
    };

    struct IsEntryLessEQ : thrust::unary_function<Entry, bool> {
        inline IsEntryLessEQ(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val <= _filter;
        }
    private:
        Entry _filter;
    };

    struct EntryKeyMatch : thrust::unary_function<Entry, bool>{
        inline EntryKeyMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val.key[0] == _filter.key[0] && val.key[1] == _filter.key[1] && _filter.valType == val.valType;
        }
    private:
        Entry _filter;
    };

    struct EntryValMatch : thrust::unary_function<Entry, bool>{
        inline EntryValMatch(const Entry & filter):_filter(filter){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val.valType == _filter.valType && val.data.bigVal == _filter.data.bigVal;
        }
    private:
        Entry _filter;
    };

    struct MatchEntryByID : thrust::unary_function<Entry, bool>{
        inline MatchEntryByID(const unsigned long long int id):_id(id){}

        __device__ __host__
        inline bool operator()(const Entry & val)const{
            return val.id == _id;
        }
    private:
        const unsigned long long int _id;
    };

}
#endif //GPU_NO_SQL_ENTRYCOMPARATORS_H
