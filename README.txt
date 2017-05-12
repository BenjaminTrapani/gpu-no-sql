GPUDB API

////////////////////////////////////////////////////////////
Run Instructions:
Modify deploy.py lines 9-10 to contain the target hostname and ssh credentials
python deploy.py
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
API Documentation (Error Codes Below)
////////////////////////////////////////////////////////////

int getRootDoc()
    Returns the root doc ID (always 0)

int getDoc(std::vector<std::string> & strings)
    Takes a list of strings representing the path to the desired doc top down
    Returns the reference ID to a given doc or an error code
        Errors: -1, -2, -5, -9

int deleteDocRef(int docID)
    Takes a document reference
    returns an error code as follows
        Errors: 0, -2, -4

int newDoc(int docID, std::string & key)
    Takes a document reference to create the new doc in and the key of the desired doc to create
    Returns the reference ID to the given doc or an error code
        Errors: -1, -2, -5, -9

int addToDoc(int docID, std::string & key, GPUDB_Value & value, GPUDB_Type & type)
    Takes a document reference, a key, a value, and a type
    Inserts the key/value pair into the given document and returns an error code
        Errors: 0, -2, -5


int batchAdd(int docID, std::vector<std::string> & keys, std::vector<GPUDB_Value> & values, std::vector<GPUDB_Type> & types)
    Takes a document reference and vectors of key/value/type and adds them to the given doc
    Returns a success or an error code
        Errors: 0, -2, -3
        -X - invalid key on the -x+1000000 place

int newFilter(int docID)
    Takes a document reference
    Creates a new filter that filters only that document, starting at the level below it
    Returns the filter reference or an error code
        Errors: -1, -2

int addToFilter(int filterID, std::string key);
    Takes a filter reference and a key to filter by at the current level
    Returns a success or error code
        Errors: 0, -5, -6

int addToFilter(int filterID, std::string key, GPUDB_Value & value, GPUDB_Type & type, GPUDB_COMP comp)
    Takes a filter reference, a key, a value, a type, and a comparator
    Adds this to the current filter level
    Returns a success or error code
        Errors: 0, -5, -6

int advanceFilter(int filterID)
    Take a filter reference
    Move the filter to the next level
        Errors: -6

int deleteFilter(int filterID)
    Take a filter reference
    Remove the filter reference
        Errors: -6

std::vector<GPUDB_QueryResult> query(int filterID)
    Run the given filter and get the result

int updateOnDoc(int filterID, GPUDB_Value & value, GPUDB_Type & type)
    takes a filter reference, a value to update to, and an updated type
    updates the given KV, error's if the filter returns a document
        Errors: -7

int deleteFromDoc(int filterID)
    takes a filter reference
    runs the filter and deletes all results and children of any results

////////////////////////////////////////////////////////////
Error Codes:
0 - success
1 - no space
2 - invalid doc reference
3 - lists are not of equal size
4 - cannot remove root
5 - Invalid Key
6 - invalid filter reference
7 - filter returns a document when it should not
9 - bad path to document
////////////////////////////////////////////////////////////
