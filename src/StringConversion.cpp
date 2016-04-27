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

    const char *srcString = src.c_str();

    char *srcBits = (char*) calloc(MAX_STRING_SIZE, sizeof(char));

    memcpy(srcBits, srcString, sizeof(char) * src.size()+1);

    memcpy(dest, srcBits, sizeof(char) * MAX_STRING_SIZE);
    return 0;
}

std::string StringConversion::intToString(const long long int *src) {
    char *endStr = (char*) calloc(MAX_STRING_SIZE, sizeof(char));

    memcpy(endStr, src, sizeof(char) * MAX_STRING_SIZE);

    return std::string(endStr);
}
