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

    bool operator==(const CodebookElement& other) {
        if (x != other.x || y != other.y) {
            return false;
        }
        if (transform != other.transform) {
            return false;
        }
        
        return true;
    }

    bool operator!=(const CodebookElement& other) {
      return !(*this == other);
    }    
};

struct RangeBlockInfo {
    // Top left coordinates in image
    int x;
    int y;
    CodebookElement* codebookElement;
    int brightnessOffset;
    float contrastFactor;

    bool operator==(const RangeBlockInfo& other) {
        if (x != other.x || y != other.y) {
            return false;
        }
        if (brightnessOffset != other.brightnessOffset || contrastFactor != other.contrastFactor) {
            return false;
        }
        if (codebookElement == NULL && other.codebookElement == NULL) {
            return true;
        } else if (codebookElement == NULL || other.codebookElement == NULL) {
            return false;
        }

        return *codebookElement == *(other.codebookElement);
    }

    bool operator!=(const RangeBlockInfo& other) {
      return !(*this == other);
    }
};

struct CompressedImage {
    int width;
    int height;
    int rangeSize;
    int domainSize;
    std::vector<RangeBlockInfo> rangeInfo;

    bool operator==(const CompressedImage& other) {
        if (width != other.width || height != other.height) {
            return false;
        }
        if (rangeSize != other.rangeSize || domainSize != other.domainSize) {
            return false;
        }
        if (rangeInfo.size() != other.rangeInfo.size()) {
            return false;
        }

        for (unsigned int i = 0; i < rangeInfo.size(); ++i) {
            if (rangeInfo[i] != other.rangeInfo[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const CompressedImage& other) {
      return !(*this == other);
    }
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