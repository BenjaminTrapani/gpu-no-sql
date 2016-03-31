//
// Created by Benjamin Trapani on 3/30/16.
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "DBElement.h"
#include "QueryResult.h"

void create(const DBElement * object);
//caller is responsible for freeing memory.
QueryResult * query(const DBElement * object);
void update(const DBElement * object);
void deleteEntry(const DBElement * object);
void sort(const DBElement * sortFilter, const DBElement * searchFilter);

#endif //SRC_GPUDBDRIVER_H
