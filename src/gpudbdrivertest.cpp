//
// Driver tests
//

#include "gpudbdrivertest.hpp"
#include "gpudbdriver.hpp"
#include "EntryUtils.hpp"
#include "Filter.hpp"

using namespace GPUDB;

void GPUDBDriverTest::runTests(){
    GPUDBDriver driver;
    printf("sizeof entry = %i\n", sizeof(Entry));
    Doc coreDoc;
    Entry lastEntry;
    Entry realLastEntry;
    for (unsigned int i = 0; i < driver.getTableSize()-3; i++) {
        Entry anEntry;
        anEntry.data.bigVal=0;
        anEntry.valType = GPUDB_BGV;
        EntryUtils::assignKeyToEntry(anEntry, i);
        anEntry.id = i;
        Doc * perm = coreDoc.addChild(anEntry);
        if(i == 3) {
            lastEntry.valType = GPUDB_BGV;
            lastEntry.data.bigVal = 1;
            EntryUtils::assignKeyToEntry(lastEntry, 10);
            lastEntry.parentID = 3;
            perm->children.push_back(lastEntry);
        } else if (i == 6) {
            realLastEntry.valType = GPUDB_BGV;
            realLastEntry.id = 51;
            realLastEntry.data.bigVal = 1;
            EntryUtils::assignKeyToEntry(realLastEntry, 10);
            realLastEntry.parentID = 6;
            perm->children.push_back(realLastEntry);
        }
    }
    clock_t t1, t2;
    t1 = clock();
    driver.create(coreDoc);
    driver.syncCreates();
    t2 = clock();
    printf("Database has %i entries\n", driver.getNumEntries());
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Creating all entries took %fms\n", diff);

    Entry filter1 = realLastEntry;
    Entry filter2;
    filter2.data.bigVal=0;
    filter2.valType = GPUDB_BGV;
    EntryUtils::assignKeyToEntry(filter2, realLastEntry.parentID);

    FilterGroup filters1;
    filters1.resultMember = false;
    FilterGroup filters2;
    filters2.resultMember = true;
    Filter concreteFilter1(filter1, EQ);
    Filter concreteFilter2(filter2, EQ);
    filters1.group.push_back(concreteFilter1);
    filters2.group.push_back(concreteFilter2);

    FilterSet filterSet;
    filterSet.push_back(filters2);
    filterSet.push_back(filters1);

    t1 = clock();
    std::vector<Doc> hostqueryResult = driver.getDocumentsForFilterSet(filterSet);
    t2 = clock();

    float diff1 = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("device multi-filter query latency = %fms\n", diff1);

    for(std::vector<Doc>::iterator iter = hostqueryResult.begin(); iter != hostqueryResult.end(); ++iter) {
        printf("Doc id = %llu\n", iter->kvPair.id);
        for(std::list<Doc>::iterator nestedIter = iter->children.begin(); nestedIter != iter->children.end();
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
            filterGroup1.group.push_back(Filter(iter->kvPair, EQ));
            filterGroup1.resultMember = false;
            FilterGroup filterGroup2;
            filterGroup2.group.push_back(Filter(newEntry, EQ));
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
    searchForLastEntry.group.push_back(Filter(lastEntry, EQ));
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

    for(size_t i = 1; i < driver.getTableSize(); i+=17){
        Doc root;
        EntryUtils::assignKeyToEntry(root.kvPair, i);
        root.kvPair.data.bigVal=i;
        root.kvPair.valType=GPUDB_BGV;
        root.kvPair.id = i;
        root.kvPair.parentID = 0;
        Doc * curParent = &root;
        for (int j = 1; j < 16; j++) {
            Entry curVal;
            EntryUtils::assignKeyToEntry(curVal, i + j);
            curVal.valType = GPUDB_BGV;
            curVal.data.bigVal = i + j;
            curVal.id = i + j;

            Doc intermediate(curVal);
            Doc * permIntermediate = curParent->addChild(intermediate);
            curParent = permIntermediate;
        }
        driver.create(root);
    }
    driver.syncCreates();
    printf("Database has %i entries\n", driver.getNumEntries());

    FilterSet filterByFirstFourNest;
    filterByFirstFourNest.reserve(4);
    for (int i = 1; i < 5; i++) {
        Entry curFilter;
        EntryUtils::assignKeyToEntry(curFilter, i);
        curFilter.valType=GPUDB_BGV;
        curFilter.data.bigVal = i;
        FilterGroup curGroup;
        if (i == 4) {
            curGroup.resultMember = true;
        }
        Filter theFilter(curFilter, EQ);
        curGroup.group.push_back(theFilter);
        filterByFirstFourNest.push_back(curGroup);
    }

    clock_t t1, t2;

    t1 = clock();
    std::vector<Doc> result = driver.getDocumentsForFilterSet(filterByFirstFourNest);
    t2 = clock();
    float diff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Deep filter took %fms\n", diff);
    printf("Num results = %i\n", result.size());
    for (std::vector<Doc>::iterator iter = result.begin(); iter != result.end(); ++iter) {
        printf(iter->toString().c_str());
    }

    printf("Deep nesting test finished.\n\n");
}

int main(int argc, char * argv[]){
    GPUDBDriverTest test;
    test.runTests();
    return 0;
}

