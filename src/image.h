#ifndef  __IMAGE_H__
#define  __IMAGE_H__

#include <iostream>

struct Image {

    Image(int w, int h) {
        width = w;
        height = h;
        data = new int[4 * width * height];
        xOffset = 0;
        yOffset = 0;
    }

    Image(int x, int y, int w, int h, int* data) {
        xOffset = x;
        yOffset = y;
        width = w;
        height = h;
        this->data = data;
    }

    void clear(int r, int g, int b, int a) {

        int numPixels = width * height;
        int* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr[3] = a;
            ptr += 4;
        }
    }

    void set(int x, int y, int r, int g, int b, int a) {
        x += xOffset;
        y += yOffset;
        int* ptr = data + ((width * y) + x) * 4;
        ptr[0] = r;
        ptr[1] = g;
        ptr[2] = b;
        ptr[3] = a;
    }

    void get(int x, int y, int* r, int* g, int* b, int* a) const {
        x += xOffset;
        y += yOffset;
        int* ptr = data + ((width * y) + x) * 4;
        *r = ptr[0];
        *g = ptr[1];
        *b = ptr[2];
        *a = ptr[3];
    }

    Image* resize(int newW, int newH) {
        Image* im = new Image(newW, newH);

        float xScale = ((float) width) / newW;
        float yScale = ((float) height) / newH;

        int r, g, b, a;
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

    int dist(Image* other, int channel) {
        if (width != other->width || height != other->height) {
            std::cerr << "Image sizes do not match" << std::endl;
            return -1;
        }

        int d = 0;
        int colors[4];
        int otherColors[4];
        int diff;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                this->get(x, y, colors, colors+1, colors+2, colors+3);
                other->get(x, y, otherColors, otherColors+1, otherColors+2, otherColors+3);
                diff = colors[channel] - otherColors[channel];
                d += diff * diff;
            }
        }
        return d;
    }

    int width;
    int height;
    int xOffset;
    int yOffset;
    int* data;
};


#endif
