#include <stdio.h>
#include <iostream>

#include "refCompressor.h"
#include "ppm.h"
#include "image.h"

RefCompressor::RefCompressor(const std::string& imageFilename, int rangeSize, int domainSize) {
    image = readPPMImage(imageFilename.c_str());
    std::cout << image->data[0] << std::endl;
    Image* im2 = image->resize(256, 256);
    writePPMImage(im2, "output.ppm");
    this->imageFilename = imageFilename;
}

RefCompressor::~RefCompressor() {
    if (image) {
        delete image;
    }
}

void RefCompressor::compress() {
    /*if (!image || image->width % rangeSize || image->height % rangeSize ||
        image->width % domainSize || image->height % domainSize) {
        std::cerr << "Invalid compression request" << std::endl;
        return;
    }

    for (int x = 0; x < image->width; x += domainSize) {
        for (int y = 0; y < image->height; y += domainSize) {
            
        }
    }*/
}

void RefCompressor::saveToFile(const std::string& filename) {

}

std::string RefCompressor::getCompressedContents() {
    return "";
}