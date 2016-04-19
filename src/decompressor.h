#ifndef __DECOMPRESSOR_H__
#define __DECOMPRESSOR_H__

#include <string>
#include "image.h"

class Decompressor {

public:
    virtual ~Decompressor() {};
    virtual void decompress() = 0;
    virtual void step() = 0;
    virtual Image* getImage() = 0;
};

#endif