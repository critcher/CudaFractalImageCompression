#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>

#include "image.h"
#include "util.h"



void readPPMImage(const char *filename, const Image* image) {
    std::ifstream f(filename);

    std::string magic;
    long width = 0;
    long height = 0;
    long pixMax = 0;

    f >> magic >> width >> height >> pixMax;

    if (f && magic == "P3" && width > 0 && height > 0 && pixMax > 0 && pixMax <= 255) {
        image = new Image(width, height);
    } else {
        std::cout << "Invalid PPM image" << std::endl;
        return;
    }

    int r, g, b;
    int numPixels = width * height;
    float* ptr = image->data;
    for (int i = 0; i < numPixels; i++) {
        f >> r >> g >> b;
        if(!f) {
            std::cout << "Invalid PPM image" << std::endl;
            return;
        }
        ptr[0] = r;
        ptr[1] = g;
        ptr[2] = b;
        ptr[3] = 1;
        ptr += 4;
    }
}

// writePPMImage --
//
// assumes input pixels are float4
// write 3-channel (8 bit --> 24 bits per pixel) ppm
void writePPMImage(const Image* image, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d ", image->width, image->height);
    fprintf(fp, "255\n");

    for (int j=image->height-1; j>=0; j--) {
        for (int i=0; i<image->width; i++) {

            const float* ptr = &image->data[4 * (j*image->width + i)];

            char val[3];
            val[0] = static_cast<char>(255.f * CLAMP(ptr[0], 0.f, 1.f));
            val[1] = static_cast<char>(255.f * CLAMP(ptr[1], 0.f, 1.f));
            val[2] = static_cast<char>(255.f * CLAMP(ptr[2], 0.f, 1.f));

            fputc(val[0], fp);
            fputc(val[1], fp);
            fputc(val[2], fp);
        }
    }

    fclose(fp);
    printf("Wrote image file %s\n", filename);
}
