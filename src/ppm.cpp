#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>

#include "image.h"
#include "util.h"



Image* readPPMImage(const char *filename) {
    std::ifstream f(filename);

    std::string magic;
    long width = 0;
    long height = 0;
    long pixMax = 0;

    f >> magic >> width >> height >> pixMax;

    Image* image;
    if (f && magic == "P3" && width > 0 && height > 0 && pixMax > 0 && pixMax <= 255) {
        image = new Image(width, height);
    } else {
        std::cerr << "Invalid PPM image" << std::endl;
        return NULL;
    }

    int r, g, b;
    int numPixels = width * height;
    float* ptr = image->data;
    for (int i = 0; i < numPixels; i++) {
        f >> r >> g >> b;
        if(!f) {
            std::cerr << "Invalid PPM image" << std::endl;
            return NULL;
        }
        ptr[0] = r;
        ptr[1] = g;
        ptr[2] = b;
        ptr[3] = 255;
        ptr += 4;
    }
    return image;
}

// writePPMImage --
//
// assumes input pixels are float4
// write 3-channel (8 bit --> 24 bits per pixel) ppm
void writePPMImage(const Image* image, const char *filename)
{
    std::ofstream f(filename);

    std::string sep = " ";

    f << "P3" << sep << image->width << sep << image->height << sep << 255 << sep << "\n";

    int r, g, b;
    int numPixels = image->width * image->height;
    float* ptr = image->data;
    for (int i = 0; i < numPixels; i++) {
        r = ptr[0];
        g = ptr[1];
        b = ptr[2];
        
        f << r << sep << g << sep << b << "\n";
        
        ptr += 4;
    }
}
