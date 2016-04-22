Work Needed Summmary

Adrian Work:
API Work
    FilterMap implementation
        switch the open spots vector to a list
    DocMap implementation
        Implement filter creation with comparators Driver expansion
        Better return id once used using a random swap
        Switch the open spots vector to a list
    Detailed Implementations: 4
        addtoFilter() - 2
        Populate Result Format in query() - 2
            Create Query Type
        flattenDoc() - 1
    Debugging - Try to Compile this shit
    Extract Magic Numbers into Constants and use on construction
    Error Code Documentation: 10

Ben Work:
    Driver Changes
        Delete
            Take a list of entries and delete all of them - makes it batch, will run fast on all deletes of all sizes
        Comparators
            Add a Comparator to each FilterGroup
                NEEDED ASAP to implement many features correctly
                Includes KEY_ONLY and VAL_ONLY
    Doc Changes
        Make the children a list instead of a vector

Work Schedule

Adrian:
Friday: API Final Sprint Day
Saturday: API Finalizations, Code Documentation and Cleanup
Sunday: Cleanup Finalizations, Powerpoint Design

Ben:
Friday: Driver Changes and Doc Update
Saturday: Write Demo Tests
Sunday: Finalize Tests, Measure Preformance

Both:
Saturday Meeting: Run Tests, Plan
Sunday Meeting: Review Powerpoint