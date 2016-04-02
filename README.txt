// Driver API

// Creates the element at the given id / index and returns an error code
int create(unsigned long long int id, char *key, GPUDB_Type type, GPUDB_Data data, unsigned long long int parentID);

// Updates the given id / index to the type/value and returns an error code
int update(unsigned long long int id, GPUDB_Type type, GPUDB_Data data);

// Deletes the data at the given id / index and returns an error code
int delete(unsigned long long int id);

// Searches the entire database for all matches to both key and value and returns a list of parent ID's
// Done on the GPU
std::vector<int> keyValueFilterGPU(char *key, GPUDB_Type type, GPUDB_Data data);

// Searches the given ids for all matches to both key and a document
// Done on the GPU
std::vector<int> keyIDFilterGPU(char *key, std::vector<int> ids);

// Searches the entire database for all matches to both key and value and returns a list of parent ID's
// Done on the CPU
std::vector<int> keyValueFilterCPU(char *key, GPUDB_Type type, GPUDB_Data data);

// Searches the given ids for all matches to both key and a document
// Done on the CPU
std::vector<int> keyIDFilterCPU(char *key, std::vector<int> ids);

// Get all needed entries for the given ids
std::vector<GPUDB_QueryResult> getEntries(std::vector<int> ids);