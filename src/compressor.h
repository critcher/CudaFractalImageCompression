#ifndef __COMPRESSOR_H__
#define __COMPRESSOR_H__

#include <string>

class Compressor {

public:
    virtual ~Compressor() {};
    virtual void compress() = 0;
    virtual void saveToFile(const std::string& filename) = 0;
    virtual std::string getCompressedContents() = 0;
};

#endif