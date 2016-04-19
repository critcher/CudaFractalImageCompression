#ifndef __FRACTAL_H__
#define __FRACTAL_H__

#include "image.h"

enum Transform {identity, rot90, rot180, rot270, flip, frot90, frot180, frot270};

struct CodebookElement {
    // Top left coordinates in image
    int x;
    int y;
    Image* imChunk;
    Transform transform;
};

#endif