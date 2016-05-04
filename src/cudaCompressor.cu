#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaCompressor.h"
#include "image.h"
#include "ppm.h"
#include "compressedFile.h"

struct GlobalConstants {
    int imageWidth;
    int imageHeight;
    int rangeSize;
    int domainSize;
    int* imageData;
};

__constant__ GlobalConstants deviceConstants;

#include "imageUtils.cu_inl"
#include "transforms.cu_inl"

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
  int x = index % w;
  int y = index / w;

  int r, g, b, a;
  int oldX = scale * x;
  int oldY = scale * y;

  pixelGet(oldX, oldY, deviceConstants.imageWidth, deviceConstants.imageHeight, &r, &g, &b, &a, deviceConstants.imageData);
  pixelSet(x, y, w, h, r, g, b, a, resizedImg);
}

__global__ void transformKernel(int* fullImg, float scale, int widthInBlocks, int* codebookElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = (index / (8 * widthInBlocks)) * deviceConstants.rangeSize;
  int domainIndex = (index / 8);
  int y = (domainIndex % widthInBlocks) * deviceConstants.rangeSize;
  int transform = index % 8;
  int* myElement = codebookElements + index * (4 * deviceConstants.rangeSize * deviceConstants.rangeSize);

  switch (transform) {
    case identity:
        identityTransform(x, y, fullImg, scale, myElement);
        break;
    case rot90:
        rot90Transform(x, y, fullImg, scale, myElement);
        break;
    case rot180:
        rot180Transform(x, y, fullImg, scale, myElement);
        break;
    case rot270:
        rot270Transform(x, y, fullImg, scale, myElement);
        break;
    case flip:
        flipTransform(x, y, fullImg, scale, myElement);
        break;
    case frot90:
        frot90Transform(x, y, fullImg, scale, myElement);
        break;
    case frot180:
        frot180Transform(x, y, fullImg, scale, myElement);
        break;
    case frot270:
        frot270Transform(x, y, fullImg, scale, myElement);
        break;
  }
}

__global__ void distanceKernel(int* codebookElements, int numCodebookeElements, int widthInRangeBlocks,
                               int* distances, int* brightnesses, float* contrasts) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int rangeNum = index / numCodebookeElements;
  int rSize = deviceConstants.rangeSize;
  int rangeX = (rangeNum % widthInRangeBlocks) * rSize;
  int rangeY = (rangeNum / widthInRangeBlocks) * rSize;
  int codebookNum = index % numCodebookeElements;
  int* codebookPtr = codebookElements + codebookNum * (4 * deviceConstants.rangeSize * deviceConstants.rangeSize);

  // Get contrast scaling factor
  int domNorm = imDot(codebookPtr, 0, 0, rSize, rSize, codebookPtr, rSize, rSize, rSize, rSize, 0);
  float con;
  if (domNorm == 0) {
      con = 0;
  } else {
      float numerator = imDot(deviceConstants.imageData, rangeX, rangeY, deviceConstants.imageWidth,
                              deviceConstants.imageHeight, codebookPtr, rSize, rSize, rSize, rSize, 0);
      con = numerator / domNorm;
  }
  
  // Calculate brightness offset
  int rangeBrightness = averageBrightness(deviceConstants.imageData, rangeX, rangeY, deviceConstants.imageWidth,
                                          deviceConstants.imageHeight, rSize, rSize, 0);
  int codebookBrightness = averageBrightness(codebookPtr, 0, 0, rSize, rSize, rSize, rSize, 0);
  int bright = rangeBrightness - con * codebookBrightness;
  
  // Calculate distance
  int dist = 0;
  int colors[4];
  int otherColors[4];
  for (int y = 0; y < rSize; y++) {
      for (int x = 0; x < rSize; x++) {
          pixelGet(x + rangeX, y + rangeY, deviceConstants.imageWidth, deviceConstants.imageHeight,
                   colors, colors+1, colors+2, colors+3, deviceConstants.imageData);
          pixelGet(x, y, rSize, rSize, otherColors, otherColors+1, otherColors+2, otherColors+3, codebookPtr);
          otherColors[0] = con * otherColors[0] + bright;
          int diff = colors[0] - otherColors[0];
          dist += diff * diff;
      }
  }

  distances[index] = dist;
  brightnesses[index] = bright;
  contrasts[index] = con;
}

CudaCompressor::CudaCompressor(const std::string& imageFilename, int rangeSize, int domainSize) {
  image = readPPMImage(imageFilename.c_str());
  this->compIm.rangeSize = rangeSize;
  this->compIm.domainSize = domainSize;
  this->compIm.width = image->width;
  this->compIm.height = image->height;

  cudaMalloc(&(cudaImageData), sizeof(int) * 4 * image->width * image->height);

  GlobalConstants hostConstants;
  hostConstants.imageWidth = image->width;
  hostConstants.imageHeight = image->height;
  hostConstants.rangeSize = rangeSize;
  hostConstants.domainSize = domainSize;
  hostConstants.imageData = cudaImageData;

  cudaMemcpy(hostConstants.imageData, image->data, sizeof(int) * 4 * image->width * image->height, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceConstants, &hostConstants, sizeof(GlobalConstants));
}

CudaCompressor::~CudaCompressor() {
  if (image) {
    delete image;
    cudaFree(cudaImageData);
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
  resizeKernel<<<resizeDim, rangeDim>>>(smallImg, scale, newW, newH);
  cudaThreadSynchronize();

  // Make Codebook elements
  int* codebookElements;
  int numDomainBlocks = (image->width / compIm.domainSize) * (image->height / compIm.domainSize);
  cudaMalloc(&(codebookElements), sizeof(int) * 4 * compIm.rangeSize * compIm.rangeSize * numDomainBlocks * 8);
  dim3 baseDim(1024, 1);
  dim3 transformDim((numDomainBlocks * 8) / baseDim.x);
  transformKernel<<<transformDim, baseDim>>>(smallImg, 1 / scale, image->width / compIm.domainSize, codebookElements);
  cudaThreadSynchronize();

  // Calculate range block-codebook element pairwise distances
  int* distances;
  int* brightnesses;
  float* contrasts;
  int numCodebookeElements = numDomainBlocks * 8;
  int numRangeBlocks = (image->width / compIm.rangeSize) * (image->height / compIm.rangeSize);
  cudaMalloc(&(distances), sizeof(int) * numCodebookeElements * numRangeBlocks);
  cudaMalloc(&(brightnesses), sizeof(int) * numCodebookeElements * numRangeBlocks);
  cudaMalloc(&(contrasts), sizeof(float) * numCodebookeElements * numRangeBlocks);
  dim3 distDim((numCodebookeElements * numRangeBlocks) / baseDim.x);
  distanceKernel<<<distDim, baseDim>>>(codebookElements, numCodebookeElements, image->width / compIm.rangeSize,
                                       distances, brightnesses, contrasts);
  cudaThreadSynchronize();

  cudaFree(smallImg);
  cudaFree(codebookElements);
  cudaFree(distances);
  cudaFree(brightnesses);
  cudaFree(contrasts);
}

void CudaCompressor::saveToFile(const std::string& filename) {
  writeFracFile(compIm, filename.c_str());
}

CompressedImage* CudaCompressor::getCompressedContents() {
    return &compIm;
}