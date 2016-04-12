// Driver API

GPUDBDriver();
        ~GPUDBDriver();

        void create(const CoreTupleType &object);

        QueryResult getRootsForFilterSet(const std::vector<CoreTupleType>& filters);
        Doc getEntriesForRoots(const HostVector_t& roots, const size_t numRoots);

        Doc getDocumentForFilterSet(const std::vector<CoreTupleType> & filters);

        void update(const CoreTupleType &searchFilter, const CoreTupleType &updates);
        void deleteBy(const CoreTupleType &searchFilter);
        void sort(const CoreTupleType &sortFilter, const CoreTupleType &searchFilter);

        inline size_t getTableSize()const{
            return numEntries;
        }