
// API Tests for GPUDB

#include "gpudb-api.hpp"
#include <cstdio>

// ********************************************************************************
// Testing

void printValue(GPUDB_Value val, GPUDB_Type type) {
    printf("Value: ");
    if (type == GPUDB_BLN) {
        printf("%d", val.b);
    } else if (type == GPUDB_INT) {
        printf("%d", val.n);
    } else if (type == GPUDB_FLT) {
        printf("%f", val.f);
    } else if (type == GPUDB_CHAR) {
        printf("%c", val.c);
    } else if (type == GPUDB_STR) {
        printf("%s", val.s);
    } else {
        printf("%lld", val.bigVal);
    }
    printf("\n");
}

void printIndent(int indent) {
    for (int i = 0; i < indent; i += 1) {
        printf(" ");
    }
}

void printResults(GPUDB_QueryResult result, int indent) {
    GPUDB_KV *kv = result.kv;
    if (kv->key.size() != 0) {
        GPUDB_KV *kv = result.kv;
        printIndent(indent);
        printf("Key: %s, ", kv->key.c_str());
        printIndent(indent);
        printValue(kv->value, kv->type);

        // Handle the children
        if (!result.children.empty()) {
            for (std::list<QueryResult>::iterator it = result.children.begin(); it != result.children.end(); it++) {
                printResults(*it, indent+2);
            }
        }
    } else {
        printf("No results\n");
    }
}

int main() {
    printf("Running main in top level API\n");
    GPUDB_Database database;

    clock_t t1, t2;
    float timeDiff;

    std::string testName("test");
    std::string testName2("test2");

    GPUDB_Type intType = GPUDB_INT;
    GPUDB_Type docType = GPUDB_DOC;
    GPUDB_Value testVal;
    testVal.n = 1;

    int rootID = database.getRootDoc();

    printf("Number of entries: %d\n", database.driver.getNumEntries());

    int firstDoc = database.newDoc(rootID, testName);
    printf("new created doc ID: %d\n", firstDoc);

    printf("Number of entries: %d\n", database.driver.getNumEntries());

    int res = database.addToDoc(firstDoc, testName2, testVal, intType);
    printf("Result of add: %d\n", res);

    printf("Number of entries: %d\n", database.driver.getNumEntries());

//    int resTotal = 0;
//    for (int i = 0; i < 10; i += 1) {
//        testVal.n += 1;
//        resTotal += database.addToDoc(firstDoc, testName2, testVal, intType);
//    }
//    printf("Many adds error total: %d", resTotal);

    int newFilter = database.newFilter(firstDoc);
    //database.addToFilter(newFilter, testName);

    t1 = clock();
    printf("\n\nResults: \n");
    printResults(database.query(newFilter), 0);
    t2 = clock();
    timeDiff = ((float)(t2 - t1) / 1000000.0F ) * 1000;
    printf("Time Taken For Query: %fms.\n", timeDiff);

    printf("Number of entries: %d\n", database.driver.getNumEntries());

    return 0;
}