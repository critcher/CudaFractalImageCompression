#include <stdio.h>

#include "refCompressor.h"
#include "image.h"

RefCompressor::RefCompressor(const std::string& imageFilename) {
    // TODO Load the image
    image = NULL;
    this->imageFilename = imageFilename;
}

RefCompressor::~RefCompressor() {
    if (image) {
        delete image;
    }
}

void RefCompressor::compress() {

}

void RefCompressor::saveToFile(const std::string& filename) {

}

std::string RefCompressor::getCompressedContents() {
    return "";
}