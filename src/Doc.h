//
// Created by Benjamin Trapani on 4/12/16.
//

#ifndef GPU_NO_SQL_DOC_H
#define GPU_NO_SQL_DOC_H
#include <vector>
#include <string>
#include <sstream>
#include "Entry.h"

namespace GPUDB {
    class Doc {
    public:
        Doc(){}
        Doc(const Entry & entry):kvPair(entry){}

        Doc * addChild(const Doc & child){
            children.push_back(child);
            children.at(children.size()-1).kvPair.parentID = kvPair.id;
            return &children[children.size()-1];
        }

        std::string toString(){
            std::stringstream sstream;
            sstream << "id: " << kvPair.id << std::endl;
            sstream << "  key: " << kvPair.key << std::endl;
            sstream << "  value: " << kvPair.data.bigVal << std::endl;
            sstream << "  parentid: " << kvPair.parentID << std::endl;
            for(std::vector<Doc>::iterator iter = children.begin(); iter != children.end(); ++iter){
                sstream << "  child " << std::distance(children.begin(), iter) << std::endl;
                sstream << iter->toString();
            }
            return sstream.str();
        }

        Entry kvPair;
        std::vector<Doc> children;
    };
}

#endif //GPU_NO_SQL_DOC_H
