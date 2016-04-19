#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaCompressor.h"
#include "image.h"

struct GlobalConstants {
    int imageWidth;
    int imageHeight;
    float* imageData;
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

CudaCompressor::CudaCompressor(const std::string& imageFilename) {
    // TODO Load the image
    image = NULL;
    this->imageFilename = imageFilename;
}

CudaCompressor::~CudaCompressor() {
    if (image) {
        delete image;
    }
}

void CudaCompressor::compress() {

}

void CudaCompressor::saveToFile(const std::string& filename) {

}

std::string CudaCompressor::getCompressedContents() {
    return "";
}