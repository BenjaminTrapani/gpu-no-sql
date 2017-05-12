//
// Created by Adrian Kant on 4/22/16.
//

#ifndef GPU_NO_SQL_STRINGCONVERSION_HPP
#define GPU_NO_SQL_STRINGCONVERSION_HPP

#include "presets.hpp"
#include <string>

namespace GPUDB {
    class StringConversion {
    public:
        static int stringToInt(long long int *dest, const std::string & src);
        static std::string intToString(const long long int *src);
    };
}

#endif //GPU_NO_SQL_STRINGCONVERSION_HPP
