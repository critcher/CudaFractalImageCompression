#ifndef __CUDA_COMPRESSOR_H__
#define __CUDA_COMPRESSOR_H__

#include "compressor.h"
#include "image.h"


class CudaCompressor : public Compressor {

private:
    Image* image;
    std::string imageFilename;

public:
    CudaCompressor(const std::string& imageFilename);
    virtual ~CudaCompressor();
    void compress();
    void saveToFile(const std::string& filename);
    std::string getCompressedContents();
};


#endif
