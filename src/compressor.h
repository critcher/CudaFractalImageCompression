#ifndef __COMPRESSOR_H__
#define __COMPRESSOR_H__

#include <string>

#include "fractal.h"

class Compressor {

public:
    virtual ~Compressor() {};
    virtual void compress() = 0;
    virtual void saveToFile(const std::string& filename) = 0;
    virtual CompressedImage* getCompressedContents() = 0;
};

#endif