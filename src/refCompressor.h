#ifndef __REF_COMPRESSOR_H__
#define __REF_COMPRESSOR_H__

#include "compressor.h"
#include "image.h"


class RefCompressor : public Compressor {

private:
    Image* image;
    std::string imageFilename;

public:
    RefCompressor(const std::string& imageFilename, int rangeSize, int domainSize);
    virtual ~RefCompressor();
    void compress();
    void saveToFile(const std::string& filename);
    std::string getCompressedContents();
};


#endif
