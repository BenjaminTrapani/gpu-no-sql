//
// Header for DocMap class
//

#ifndef GPU_NO_SQL_DOCMAP_H
#define GPU_NO_SQL_DOCMAP_H

#include <list>
#include <string>
#include "FilterSet.hpp"
#include "gpudbdriver.hpp"
#include "presets.hpp"
#include "StringConversion.hpp"

using namespace GPUDB;

class DocMap {
public:
    DocMap(GPUDBDriver *d);
    // Returns the external id for the given doc path
    int addDoc(std::vector<std::string> strings);
    // returns the internal val for the external doc
    unsigned long long int getDoc(int docID);
    // return the paths vector
    std::vector<std::string> getPath(int docID);
    // Get the filter set to search for this doc
    unsigned long long int getFilterSet(int docID);
    // Removes the external id from the mappings and returns an exit code
    int removeDoc(int docID);
private:
    GPUDBDriver *driver;

    std::list<int> openSpots;

    unsigned long long int docs[MAX_RESOURCES];
    FilterSet filters[MAX_RESOURCES];
    std::vector<std::string> paths[MAX_RESOURCES];

    bool validID(int filterID);
};

#endif //GPU_NO_SQL_DOCMAP_H
