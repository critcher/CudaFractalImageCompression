#ifndef __FRACTAL_H__
#define __FRACTAL_H__

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

#endif