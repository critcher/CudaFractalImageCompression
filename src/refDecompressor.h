#ifndef __REF_DECOMPRESSOR_H__
#define __REF_DECOMPRESSOR_H__

#include "decompressor.h"

class RefDecompressor : public Decompressor {

private:

public:
    RefDecompressor(const std::string& compressedFilename);
    virtual ~RefDecompressor();
    void decompress();
    void step();
    Image* getImage();
};

#endif
