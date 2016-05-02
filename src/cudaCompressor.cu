#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaCompressor.h"
#include "image.h"
#include "ppm.h"
#include "compressedFile.h"

#include "imageUtils.cu_inl"

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

__global__ void resizeKernel(int* resizedImg, float scale, int w, int h) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = index % deviceConstants.imageWidth;
    int y = index / deviceConstants.imageWidth;

    int r, g, b, a;
    int oldX = scale * x;
    int oldY = scale * y;

    pixelGet(oldX, oldY, deviceConstants.imageWidth, deviceConstants.imageHeight, &r, &g, &b, &a, deviceConstants.imageData);
    pixelSet(x, y, w, h, r, g, b, a, resizedImg);
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
  if (!image || image->width % compIm.rangeSize || image->height % compIm.rangeSize ||
      image->width % compIm.domainSize || image->height % compIm.domainSize) {
    std::cerr << "Invalid compression request" << std::endl;
    return;
  }
  dim3 rangeDim(compIm.rangeSize * compIm.rangeSize, 1);

  // Get resized image
  int* smallImg;
  float scale = ((float) compIm.domainSize) / compIm.rangeSize;
  int newW = image->width / scale;
  int newH = image->height / scale;
  
  cudaMalloc(&(smallImg), sizeof(int) * 4 * newW * newH);

  dim3 resizeDim((newW * newH) / rangeDim.x);
  circleChunkTestKernel<<<resizeDim, rangeDim>>>(smallImg, scale, newW, newH);
  cudaThreadSynchronize();
}

void CudaCompressor::saveToFile(const std::string& filename) {
  writeFracFile(compIm, filename.c_str());
}

CompressedImage* CudaCompressor::getCompressedContents() {
    return &compIm;
}