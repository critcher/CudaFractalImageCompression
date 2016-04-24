#include <iostream>

#include "refDecompressor.h"
#include "compressedFile.h"

RefDecompressor::RefDecompressor(const std::string& compressedFilename) {
    readFracFile(compressedFilename.c_str(), &compIm);
    image = new Image(compIm.width, compIm.height);
    image->clear(128, 128, 128, 128);
    std::cout << "Pixel is " << image->data[0] << std::endl;
}

RefDecompressor::~RefDecompressor() {
    if (image) {
        delete image;
    }
}

void RefDecompressor::decompress() {

}

void RefDecompressor::step() {
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    Image* buffer = image->resize(scale * image->width, scale * image->height);
    int numRangeX = compIm.width / compIm.rangeSize;

    for (unsigned int i = 0; i < compIm.rangeInfo.size(); ++i) {
        int x = (i % numRangeX) * compIm.rangeSize;
        int y = (i / numRangeX) * compIm.rangeSize;
        RangeBlockInfo r = compIm.rangeInfo[i];
        Image rangeChunk(x, y, compIm.rangeSize, compIm.rangeSize, image->data, compIm.width, compIm.height);

        switch (r.codebookElement->transform) {
            case identity:
                identityTransform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case rot90:
                rot90Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case rot180:
                rot180Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case rot270:
                rot270Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case flip:
                flipTransform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case frot90:
                frot90Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case frot180:
                frot180Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
            case frot270:
                frot270Transform(r.codebookElement->x, r.codebookElement->y, buffer, scale, &rangeChunk);
                break;
        }
        rangeChunk.adjustColor(r.brightnessOffset, r.contrastFactor, 0);
    }

    delete buffer;
}

Image* RefDecompressor::getImage() {
    return image;
}