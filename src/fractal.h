#ifndef __FRACTAL_H__
#define __FRACTAL_H__

#include <vector>

#include "image.h"

enum Transform {identity=0, rot90=1, rot180=2, rot270=3, flip=4, frot90=5, frot180=6, frot270=7};

struct CodebookElement {
    // Top left coordinates in image
    int x;
    int y;
    Image* imChunk;
    Transform transform;
};

struct RangeBlockInfo {
    // Top left coordinates in image
    int x;
    int y;
    CodebookElement* codebookElement;
    int brightnessOffset;
    float contrastFactor;
};

struct CompressedImage {
    int width;
    int height;
    int rangeSize;
    int domainSize;
    std::vector<RangeBlockInfo> rangeInfo;
};

void identityTransform(int x, int y, Image* fullImg, float scale, Image* im);
void rot90Transform(int x, int y, Image* fullImg, float scale, Image* im);
void rot180Transform(int x, int y, Image* fullImg, float scale, Image* im);
void rot270Transform(int x, int y, Image* fullImg, float scale, Image* im);
void flipTransform(int x, int y, Image* fullImg, float scale, Image* im);
void frot90Transform(int x, int y, Image* fullImg, float scale, Image* im);
void frot180Transform(int x, int y, Image* fullImg, float scale, Image* im);
void frot270Transform(int x, int y, Image* fullImg, float scale, Image* im);

#endif