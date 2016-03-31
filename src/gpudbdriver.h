//
// GPUDB Driver Header (API and Helpers)
//

#ifndef SRC_GPUDBDRIVER_H
#define SRC_GPUDBDRIVER_H

#include "DBStructs.h"

// Caller must free memory

void create(const GPUDB_Element *object);

GPUDB_QueryResult * query(const GPUDB_Element *searchFilter);

void update(const GPUDB_Element *searchFilter, const GPUDB_Element *updates);

void delete(const GPUDB_Element *searchFilter);

void sort(const GPUDB_Element *sortFilter, const GPUDB_Element *searchFilter);

#endif // SRC_GPUDBDRIVER_H
