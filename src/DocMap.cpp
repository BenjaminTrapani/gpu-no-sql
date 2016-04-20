//
// TODO
//

#ifndef GPU_NO_SQL_DOCMAP_H
#define GPU_NO_SQL_DOCMAP_H

using namespace GPUDB;

class DocMap {
public:
    DocMap();
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
    std::vector<int> openSpots;
    unsigned long long int docs[1000];
    FilterSet filters[1000];
    std::vector<std::string> paths[1000];
};

DocMap::DocMap() {
    // Set Up Open Docs
    openSpots.reserve(1000);
    for (int i = 999; i > 0; i--) {
        openSpots.push_back(i);
    }
}

int DocMap::addDoc(std::vector <std::string> strings) {
    if (openSpots.size != 0) {
        // Get a place
        int place = openSpots.back();
        openSpots.pop_back();
        // TODO
        // Create filter set
        // Add it to filter set spot
        // get documentID and add it to doc spot
        return 0;
    } else {
        return -1;
    }
}

unsigned long long int DocMap::getDoc(int docID) {
    return docs[i];
}

std::vector<std::string> DocMap::getPath(int docID) {
    return paths[i];
}

unsigned long long int DocMap::getFilterSet(int docID) {
    return filters[i];
}

int DocMap::removeDoc(int docID) {
    openSpots.push_back(docID);
    // TODO better add back with random swap
    return 0;
}


#endif //GPU_NO_SQL_DOCMAP_H
