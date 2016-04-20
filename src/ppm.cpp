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
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            f >> r >> g >> b;
            if(!f) {
                std::cerr << "Invalid PPM image" << std::endl;
                return NULL;
            }
            image->set(x, y, r, g, b, 255);
        }
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

    int r, g, b, a;
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            image->get(x, y, &r, &g, &b, &a);
            f << r << sep << g << sep << b << "\n";
        }
    }
}
