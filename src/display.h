#ifndef __DISPLAY_H__
#define __DISPLAY_H__

#include "compressor.h"
#include "decompressor.h"

struct Display{
    int width;
    int height;
    bool updateDecomp;
    Compressor* compressor;
    Decompressor* decompressor;
};

void handleReshape(int w, int h);
void handleDisplay();
void handleKeyPress(unsigned char key, int x, int y);
void renderPicture();
void startCompressionWithDisplay(Compressor* compressor);


#endif