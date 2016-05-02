#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaCompressor.h"
#include "image.h"

struct GlobalConstants {
    int imageWidth;
    int imageHeight;
    int* imageData;
};

__constant__ GlobalConstants deviceConstants;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

CudaCompressor::CudaCompressor(const std::string& imageFilename, int rangeSize, int domainSize) {
  image = readPPMImage(imageFilename.c_str());
  this->compIm.rangeSize = rangeSize;
  this->compIm.domainSize = domainSize;
  this->compIm.width = image->width;
  this->compIm.height = image->height;

  GlobalConstants hostConstants;
  hostConstants.imageWidth = image->width;
  hostConstants.imageHeight = image->height;

  cudaMalloc(&(hostConstants.imageData), sizeof(int) * 4 * image->width * image->height);
  cudaMemcpy(hostConstants.imageData, image->data, sizeof(int) * 4 * image->width * image->height, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceConstants, &hostConstants, sizeof(GlobalConstants));
}

CudaCompressor::~CudaCompressor() {
  if (image) {
    delete image;
  }
}

void CudaCompressor::compress() {

}

void CudaCompressor::saveToFile(const std::string& filename) {
  writeFracFile(compIm, filename.c_str());
}

CompressedImage* CudaCompressor::getCompressedContents() {
    return &compIm;
}