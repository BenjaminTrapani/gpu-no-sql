//
// Created by Benjamin Trapani on 4/19/16.
//

#include "gpudbdrivertest.h"
#include "gpudbdriver.h"
using namespace GPUDB;

void GPUDBDriverTest::runTests(){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));
    Doc coreDoc;
    for(unsigned int i = 0; i < driver.getTableSize()-3; i++){
        Entry anEntry;
        anEntry.data.bigVal=0;
        anEntry.valType = GPUDB_BGV;
        anEntry.key=i;
        anEntry.id = i;
        coreDoc.children.push_back(Doc(anEntry));
    }
    Entry lastEntry;
    lastEntry.valType = GPUDB_BGV;
    lastEntry.data.bigVal = 1;
    lastEntry.key = 10;
    lastEntry.parentID = 3;
    coreDoc.children[3].children.push_back(lastEntry);

    Entry realLastEntry;
    realLastEntry.valType = GPUDB_BGV;
    realLastEntry.id = 51;
    realLastEntry.data.bigVal = 1;
    realLastEntry.key = 10;
    realLastEntry.parentID = 6;

    coreDoc.children[6].children.push_back(realLastEntry);

    driver.create(coreDoc);
    driver.syncCreates();
    printf("Database has %i entries\n", driver.getNumEntries());

    Entry filter1 = realLastEntry;
    Entry filter2;
    filter2.data.bigVal=0;
    filter2.valType = GPUDB_BGV;
    filter2.key=realLastEntry.parentID;

    FilterGroup filters1;
    filters1.resultMember = false;
    FilterGroup filters2;
    filters2.resultMember = true;
    filters1.group.push_back(filter1);
    filters2.group.push_back(filter2);

    FilterSet filterSet;
    filterSet.push_back(filters2);
    filterSet.push_back(filters1);

    clock_t t1, t2;
    t1 = clock();
    std::vector<Doc> hostqueryResult = driver.getDocumentsForFilterSet(filterSet);
    t2 = clock();

    float diff1 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device multi-filter query latency = %fms\n", diff1);

    for(std::vector<Doc>::iterator iter = hostqueryResult.begin(); iter != hostqueryResult.end(); ++iter){
        printf("Doc id = %llu\n", iter->kvPair.id);
        for(std::vector<Doc>::iterator nestedIter = iter->children.begin(); nestedIter != iter->children.end();
            ++nestedIter){
            printf("  child id = %llu\n", nestedIter->kvPair.id);
            Entry newEntry = nestedIter->kvPair;
            newEntry.data.bigVal = 52;
            t1 = clock();
            driver.update(nestedIter->kvPair, newEntry);
            t2 = clock();
            float diff2 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
            printf("update single element latency = %fms\n", diff2);

            FilterGroup filterGroup1;
            filterGroup1.group.push_back(iter->kvPair);
            filterGroup1.resultMember = false;
            FilterGroup filterGroup2;
            filterGroup2.group.push_back(newEntry);
            filterGroup2.resultMember = true;
            FilterSet toCheck;
            toCheck.push_back(filterGroup1);
            toCheck.push_back(filterGroup2);
            std::vector<Doc> updatedElement = driver.getDocumentsForFilterSet(toCheck);
            for(std::vector<Doc>::iterator updatedIter = updatedElement.begin();
                updatedIter != updatedElement.end(); ++updatedIter){
                printf("Updated value for id %llu = %lld\n", updatedIter->kvPair.id, updatedIter->kvPair.data.bigVal);
            }
        }
    }
    t1 = clock();
    driver.deleteBy(lastEntry);
    t2 = clock();
    float deleteDiff = ((float)(t2 - t1) / 1000000.0F ) * 1000;

    FilterGroup searchForLastEntry;
    searchForLastEntry.group.push_back(lastEntry);
    FilterSet searchForLastEntryFilter;
    searchForLastEntryFilter.push_back(searchForLastEntry);
    std::vector<Doc> lastEntryResult = driver.getDocumentsForFilterSet(searchForLastEntryFilter);
    if(lastEntryResult.size() == 0){
        printf("Successfully deleted last entry. Delete took %fms.\n", deleteDiff);
    }else{
        printf("Delete of last entry failed, still present in table.\n");
        for(std::vector<Doc>::iterator iter = lastEntryResult.begin(); iter != lastEntryResult.end(); ++iter){
            printf("Remaining id = %llu\n", iter->kvPair.id);
        }
    }

    runDeepNestingTests();
}

void GPUDBDriverTest::runDeepNestingTests(){
    printf("Beginning deep nesting test:\n");

    GPUDBDriver driver;

    for(size_t i = 2; i < driver.getTableSize(); i+=5){
        Doc root;
        root.kvPair.key=i;
        root.kvPair.data.bigVal=i;
        root.kvPair.valType=GPUDB_BGV;
        root.kvPair.id = i;
        root.kvPair.parentID = 0;
        generateNestedDoc(3, &root, i+1);
        driver.create(root);
    }
    driver.syncCreates();
    printf("Database has %i entries\n", driver.getNumEntries());

    FilterSet filterByFirstFourNest;
    filterByFirstFourNest.reserve(4);
    for(int i = 2; i < 5; i++){
        Entry curFilter;
        curFilter.key = i;
        curFilter.valType=GPUDB_BGV;
        curFilter.data.bigVal = i;
        FilterGroup curGroup;
        if(i==3){
            curGroup.resultMember = true;
        }
        curGroup.group.push_back(curFilter);
        filterByFirstFourNest.push_back(curGroup);
    }

    clock_t t1, t2;

    t1 = clock();
    std::vector<Doc> result = driver.getDocumentsForFilterSet(filterByFirstFourNest);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Deep filter took %fms\n", diff);
    printf("Num results = %i\n", result.size());
    for(std::vector<Doc>::iterator iter = result.begin(); iter != result.end(); ++iter){
        printf(iter->toString().c_str());
    }

    printf("Deep nesting test finished.\n\n");
}

void GPUDBDriverTest::generateNestedDoc(size_t nestings, Doc * parent, size_t beginIndex) {
    Entry curVal;
    curVal.key = beginIndex;
    curVal.valType = GPUDB_BGV;
    curVal.data.bigVal = beginIndex;
    curVal.id = beginIndex;

    Doc intermediate(curVal);
    Doc * permIntermediate = parent->addChild(intermediate);

    if(nestings>0)
        generateNestedDoc(nestings-1, permIntermediate, beginIndex+1);
}

int main(int argc, char * argv[]){
    GPUDBDriverTest test;
    test.runTests();
    return 0;
}

