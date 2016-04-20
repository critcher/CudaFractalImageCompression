#include <stdio.h>
#include <iostream>

#include "refCompressor.h"
#include "ppm.h"
#include "image.h"

CodebookElement RefCompressor::generateIdentity(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curX, curY, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = identity;
    return cbe;
}

CodebookElement RefCompressor::generateRotate90(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(rangeSize - curY - 1, curX, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot90;
    return cbe;
}

CodebookElement RefCompressor::generateRotate180(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(rangeSize - curX - 1, rangeSize - curY - 1, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot180;
    return cbe;
}

CodebookElement RefCompressor::generateRotate270(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curY, rangeSize - curX - 1, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot270;
    return cbe;
}

CodebookElement RefCompressor::generateFlip(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(rangeSize - curX - 1, curY, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = flip;
    return cbe;
}

CodebookElement RefCompressor::generateFRot90(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(rangeSize - curY - 1, rangeSize - curX - 1, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot90;
    return cbe;
}

CodebookElement RefCompressor::generateFRot180(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curX, rangeSize - curY - 1, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot180;
    return cbe;
}

CodebookElement RefCompressor::generateFRot270(int x, int y, Image* fullImg) {
    Image* im = new Image(rangeSize, rangeSize);
    float scale = ((float) rangeSize) / domainSize;
    int r, g, b, a;
    for (int curY = 0; curY < rangeSize; curY++) {
        for (int curX = 0; curX < rangeSize; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curY, curX, r, g, b, a);
        }
    }
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot270;
    return cbe;
}

void RefCompressor::generateCodebookEelements() {
    float scale = ((float) rangeSize) / domainSize;
    Image* smallImg = image->resize(scale * image->width, scale * image->height);

    for (int x = 0; x < image->width; x += domainSize) {
        for (int y = 0; y < image->height; y += domainSize) {
            codebook.push_back(generateFRot270(x, y, smallImg));
            codebook.push_back(generateFRot180(x, y, smallImg));
            codebook.push_back(generateFRot90(x, y, smallImg));
            codebook.push_back(generateFlip(x, y, smallImg));
            codebook.push_back(generateRotate270(x, y, smallImg));
            codebook.push_back(generateRotate180(x, y, smallImg));
            codebook.push_back(generateRotate90(x, y, smallImg));
            codebook.push_back(generateIdentity(x, y, smallImg));
        }
    }
    delete smallImg;
}

RefCompressor::RefCompressor(const std::string& imageFilename, int rangeSize, int domainSize) {
    image = readPPMImage(imageFilename.c_str());
    this->imageFilename = imageFilename;
    this->rangeSize = rangeSize;
    this->domainSize = domainSize;
}

RefCompressor::~RefCompressor() {
    if (image) {
        delete image;
    }
}

void RefCompressor::compress() {
    if (!image || image->width % rangeSize || image->height % rangeSize ||
        image->width % domainSize || image->height % domainSize) {
        std::cerr << "Invalid compression request" << std::endl;
        return;
    }

    generateCodebookEelements();
    std::cout << "generated " << codebook.size() << " codebook elements" << std::endl;
}

void RefCompressor::saveToFile(const std::string& filename) {

}

std::string RefCompressor::getCompressedContents() {
    return "";
}