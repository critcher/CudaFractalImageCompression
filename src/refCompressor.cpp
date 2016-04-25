#include <stdio.h>
#include <iostream>

#include "refCompressor.h"
#include "ppm.h"
#include "image.h"
#include "compressedFile.h"

CodebookElement RefCompressor::generateIdentity(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    identityTransform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = identity;
    return cbe;
}

CodebookElement RefCompressor::generateRotate90(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    rot90Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot90;
    return cbe;
}

CodebookElement RefCompressor::generateRotate180(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    rot180Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot180;
    return cbe;
}

CodebookElement RefCompressor::generateRotate270(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    rot270Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = rot270;
    return cbe;
}

CodebookElement RefCompressor::generateFlip(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    flipTransform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = flip;
    return cbe;
}

CodebookElement RefCompressor::generateFRot90(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    frot90Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot90;
    return cbe;
}

CodebookElement RefCompressor::generateFRot180(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    frot180Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot180;
    return cbe;
}

CodebookElement RefCompressor::generateFRot270(int x, int y, Image* fullImg) {
    Image* im = new Image(compIm.rangeSize, compIm.rangeSize);
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    frot270Transform(x, y, fullImg, scale, im);
    CodebookElement cbe;
    cbe.x = x;
    cbe.y = y;
    cbe.imChunk = im;
    cbe.transform = frot270;
    return cbe;
}

void RefCompressor::generateCodebookEelements() {
    float scale = ((float) compIm.rangeSize) / compIm.domainSize;
    Image* smallImg = image->resize(scale * image->width, scale * image->height);

    for (int x = 0; x < image->width; x += compIm.domainSize) {
        for (int y = 0; y < image->height; y += compIm.domainSize) {
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

void RefCompressor::getBestMapping() {
    int totalD = 0;
    for (int y = 0; y < image->height; y += compIm.rangeSize) {
        for (int x = 0; x < image->width; x += compIm.rangeSize) {
            Image rangeBlock(x, y, compIm.rangeSize, compIm.rangeSize, image->data, image->width, image->height);
            int minDist = 0;
            int minElement = -1;
            float bestContrast;
            int bestBrightness;
            for (unsigned int i = 0; i < codebook.size(); ++i) {

                // Just use red channel for now (assuming greyscale image)
                int domNorm = codebook[i].imChunk->dot(codebook[i].imChunk, 0);
                float con;
                if (domNorm == 0) {
                    con = 0;
                } else {
                    con = ((float) rangeBlock.dot(codebook[i].imChunk, 0)) / domNorm;
                }
                int bright = rangeBlock.getAvgBrightness(0) - con * codebook[i].imChunk->getAvgBrightness(0);
                Image* newDom = new Image(compIm.rangeSize, compIm.rangeSize);
                newDom->copyFrom(codebook[i].imChunk);
                newDom->adjustColor(bright, con, 0);
                int curDist = newDom->dist(&rangeBlock, 0);
                delete newDom;

                if (minElement < 0 || curDist < minDist) {
                    minElement = i;
                    minDist = curDist;
                    bestContrast = con;
                    bestBrightness = bright;
                }
            }
            RangeBlockInfo rb;
            rb.x = x;
            rb.y = y;
            rb.codebookElement = &(codebook[minElement]);
            rb.brightnessOffset = bestBrightness;
            rb.contrastFactor = bestContrast;
            totalD += minDist;
            compIm.rangeInfo.push_back(rb);
        }
    }
    std::cout << "Total dist: " << totalD << std::endl;
}

RefCompressor::RefCompressor(const std::string& imageFilename, int rangeSize, int domainSize) {
    image = readPPMImage(imageFilename.c_str());
    this->imageFilename = imageFilename;
    this->compIm.rangeSize = rangeSize;
    this->compIm.domainSize = domainSize;
    this->compIm.width = image->width;
    this->compIm.height = image->height;
}

RefCompressor::~RefCompressor() {
    if (image) {
        delete image;
    }
}

void RefCompressor::compress() {
    if (!image || image->width % compIm.rangeSize || image->height % compIm.rangeSize ||
        image->width % compIm.domainSize || image->height % compIm.domainSize) {
        std::cerr << "Invalid compression request" << std::endl;
        return;
    }

    generateCodebookEelements();
    std::cout << "generated " << codebook.size() << " codebook elements" << std::endl;
    getBestMapping();
}

void RefCompressor::saveToFile(const std::string& filename) {
    writeFracFile(compIm, filename.c_str());
}

CompressedImage* RefCompressor::getCompressedContents() {
    return &compIm;
}