//
// Created by Benjamin Trapani on 4/26/16.
//

#include "CPUAggregator.h"
#include <algorithm>
using namespace GPUDB;

void CPUAggregator::buildResultsWithParent(Doc * parent){
    IDToChildIDsMap_t::iterator parentPos = idToChildIdsMap.find(parent->kvPair.id);
    if(parentPos != idToChildIdsMap.end()) {
        for (std::list<unsigned long long int>::iterator iter = parentPos->second.begin();
             iter != parentPos->second.end(); ++iter) {
            Doc *perm = parent->addChild(Doc(idToEntryMap[*iter]));
            if(perm->kvPair.id)
                buildResultsWithParent(perm);
        }
    }
}

void CPUAggregator::buildResults(const thrust::host_vector<Entry> & roots, const size_t numRoots,
                                 std::vector<Doc> & results){
    size_t curIndex = 0;

    for(thrust::host_vector<Entry>::const_iterator iter = roots.begin();
        iter != roots.begin() + numRoots; ++iter){

        Doc aRoot(*iter);
        results.push_back(aRoot);
        Doc * perm = &results[curIndex];
        printf("rootid = %llu\n", perm->kvPair.id);
        buildResultsWithParent(perm);
        curIndex++;
    }
}

void CPUAggregator::onEntryCreate(const Entry &toCreate) {
    idToEntryMap[toCreate.id] = toCreate;
    idToChildIdsMap[toCreate.parentID].push_back(toCreate.id);
}

//update is not allowed to change parentid or id.
void CPUAggregator::onUpdate(const unsigned long long int id, const Entry & updatedVal){
    idToEntryMap[id].data.bigVal = updatedVal.data.bigVal;
    idToEntryMap[id].valType = updatedVal.valType;
}

void CPUAggregator::onDelete(const unsigned long long int id){
    if(idToEntryMap.find(id) != idToEntryMap.end()) {
        Entry *oldEntry = &idToEntryMap[id];
        IDToChildIDsMap_t::iterator childListPos = idToChildIdsMap.find(id);

        if (childListPos != idToChildIdsMap.end()) {
            idToChildIdsMap.erase(idToChildIdsMap.find(id));
        }

        std::list<unsigned long long int>::iterator staleChildPos =
                std::find(idToChildIdsMap[oldEntry->parentID].begin(), idToChildIdsMap[oldEntry->parentID].end(),
                          id);
        idToChildIdsMap[oldEntry->parentID].erase(staleChildPos);

        idToEntryMap.erase(idToEntryMap.find(id));
    }
}