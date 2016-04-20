#ifndef  __IMAGE_H__
#define  __IMAGE_H__

#include <iostream>

struct Image {

    Image(int w, int h) {
        width = w;
        height = h;
        data = new float[4 * width * height];
    }

    void clear(float r, float g, float b, float a) {

        int numPixels = width * height;
        float* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr[3] = a;
            ptr += 4;
        }
    }

    void set(int x, int y, float r, float g, float b, float a) {
        float* ptr = data + ((width * y) + x) * 4;
        ptr[0] = r;
        ptr[1] = g;
        ptr[2] = b;
        ptr[3] = a;
    }

    void get(int x, int y, float* r, float* g, float* b, float* a) {
        float* ptr = data + ((width * y) + x) * 4;
        *r = ptr[0];
        *g = ptr[1];
        *b = ptr[2];
        *a = ptr[3];
    }

    Image* resize(int newW, int newH) {
        Image* im = new Image(newW, newH);

        float xScale = ((float) width) / newW;
        float yScale = ((float) height) / newH;

        float r, g, b, a;
        for (int y = 0; y < newH; y++) {
            for (int x = 0; x < newW; x++) {
                int oldX = xScale * x;
                int oldY = yScale * y;
                this->get(oldX, oldY, &r, &g, &b, &a);
                im->set(x, y, r, g, b, a);
            }
        }
        return im;
    }

    int width;
    int height;
    float* data;
};


#endif
