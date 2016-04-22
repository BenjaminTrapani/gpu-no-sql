Work Needed Summmary

Driver Changes
    Delete
        Take a list of entries and delete all of them - makes it batch, will run fast on all deletes of all sizes
    Comparators
        Add a Comparator to each FilterGroup
            NEEDED ASAP to implement many features correctly
            Includes KEY_ONLY and VAL_ONLY
API Work
    FilterMap implementation
        switch the open spots vector to a list
    Detailed Implementations: 4
        addtoFilter() - 2
        Populate Result Format in query() - 2
            Create Query Type
        flattenDoc() - 1
    Extract Magic Numbers into Constants and use on construction
    Error Code Documentation: 7
    Debugging - Try to Compile this shit

Other Changes By File
    Doc
        Make the children a list instead of a vector
    DocMap
        Error Code Documentation: 3
        Implement filter creation with comparators Driver expansion
        Better return id once used using a random swap
        Switch the open spots vector to a list
