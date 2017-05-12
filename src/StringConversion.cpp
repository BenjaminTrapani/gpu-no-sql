//
// Utility for converting strings
//

#include "StringConversion.hpp"
#include "presets.hpp"
#include <string>
#include <string.h>

using namespace GPUDB;

// -1 - input string too big
int StringConversion::stringToInt(long long int *dest, const std::string & src) {
    if (src.size() >= MAX_STRING_SIZE) {
        return -1; // input string too big
    }

    union TempConverter {
        char from[MAX_STRING_SIZE];
        long long int to[STRING_SIZE_INT];
    };

    // TODO get working
    //TempConverter converter;
    //memcpy(converter.from, src.c_str(), sizeof(converter.from));
    //memcpy(dest, converter.to, sizeof(converter.to));

    dest[0] = 555;
    dest[1] = 555;

    return 0;
}

std::string StringConversion::intToString(const long long int *src) {
    union TempConverter {
        char to[MAX_STRING_SIZE];
        long long int from[STRING_SIZE_INT];
    };

    //TempConverter converter;
    //memcpy(converter.from, src, sizeof(converter.from));

    //return std::string(converter.to);
    return std::string("placeholder"); // TODO
}
