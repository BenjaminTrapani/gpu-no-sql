//
// Created by Benjamin Trapani on 4/12/16.
//

#ifndef GPU_NO_SQL_DOC_H
#define GPU_NO_SQL_DOC_H
#include <list>
#include <string>
#include <sstream>
#include "Entry.hpp"

namespace GPUDB {
    class Doc {
    public:
        Doc():parent(0){}
        Doc(const Entry & entry):kvPair(entry), parent(0){}
        Doc(const Entry & entry, Doc * iparent): kvPair(entry), parent(iparent){}

        Doc * addChild(const Doc & child){
            children.push_back(child);
            Doc * permResult = &children.back();
            permResult->parent = this;
            permResult->kvPair.parentID = kvPair.id;
            return permResult;
        }

        std::string toString(){
            std::stringstream sstream;
            sstream << "id: " << kvPair.id << std::endl;
            sstream << "  key: " << kvPair.key[0] << " " << kvPair.key[1] << std::endl;
            sstream << "  value: " << kvPair.data.bigVal << std::endl;
            sstream << "  parentid: " << kvPair.parentID << std::endl;
            for(std::list<Doc>::iterator iter = children.begin(); iter != children.end(); ++iter){
                sstream << "  child " << std::distance(children.begin(), iter) << std::endl;
                sstream << iter->toString();
            }
            return sstream.str();
        }

        Doc * parent;
        Entry kvPair;
        std::list<Doc> children;
    };
}

#endif //GPU_NO_SQL_DOC_H
