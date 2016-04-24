#include "fractal.h"

void identityTransform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curX, curY, r, g, b, a);
        }
    }
}

void rot90Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(im->height - curY - 1, curX, r, g, b, a);
        }
    }
}

void rot180Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(im->width - curX - 1, im->height - curY - 1, r, g, b, a);
        }
    }
}

void rot270Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curY, im->width - curX - 1, r, g, b, a);
        }
    }
}

void flipTransform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(im->width - curX - 1, curY, r, g, b, a);
        }
    }
}

void frot90Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(im->height - curY - 1, im->width - curX - 1, r, g, b, a);
        }
    }
}

void frot180Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curX, im->height - curY - 1, r, g, b, a);
        }
    }
}

void frot270Transform(int x, int y, Image* fullImg, float scale, Image* im) {
    int r, g, b, a;
    for (int curY = 0; curY < im->height; curY++) {
        for (int curX = 0; curX < im->width; curX++) {
            fullImg->get(curX + x * scale, curY + y * scale, &r, &g, &b, &a);
            im->set(curY, curX, r, g, b, a);
        }
    }
}