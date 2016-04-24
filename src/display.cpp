#include <algorithm>
#include <iostream>
#include <climits>

#include "display.h"
#include "image.h"
#include "platformgl.h"
#include "cycleTimer.h"
#include "refDecompressor.h"


static Display gDisplay;


// handleReshape --
//
// Event handler, fired when the window is resized
void handleReshape(int w, int h) {
    gDisplay.width = w;
    gDisplay.height = h;
    glViewport(0, 0, gDisplay.width, gDisplay.height);
    glutPostRedisplay();
}

void handleDisplay() {

    // Decompulation and rendering work is done in the renderPicture
    // function below

    renderPicture();

    // the subsequent code uses OpenGL to present the state of the
    // rendered image on the screen.

    Image* img = gDisplay.decompressor->getImage();

    int width = std::min(img->width, gDisplay.width);
    int height = std::min(img->height, gDisplay.height);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, gDisplay.width, 0.f, gDisplay.height, -1.f, 1.f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // copy image data from the compressor to the OpenGL
    // frame-buffer.  This is inefficient solution is the processing
    // to generate the image is done in CUDA.  An improved solution
    // would render to a CUDA surface object (stored in GPU memory),
    // and then bind this surface as a texture enabling it's use in
    // normal openGL rendering
    glRasterPos2i(0, 0);
    float scalar = INT_MAX / 255.0;
    glPixelTransferf(GL_RED_SCALE, scalar);
    glPixelTransferf(GL_GREEN_SCALE, scalar);
    glPixelTransferf(GL_BLUE_SCALE, scalar);
    glDrawPixels(width, height, GL_RGBA, GL_INT, img->data);

    glutSwapBuffers();
    glutPostRedisplay();
}

// handleKeyPress --
//
// Keyboard event handler
void handleKeyPress(unsigned char key, int x, int y) {
    switch (key) {
    case 'q':
    case 'Q':
        exit(1);
        break;
    case '+':
    case '=':
        gDisplay.updateDecomp = true;
        break;
    }
}

void renderPicture() {
    if (gDisplay.updateDecomp) {
        gDisplay.decompressor->step();
        gDisplay.updateDecomp = false;
    }
}

void startCompressionWithDisplay(Compressor* compressor) {
    // Save the compressed image
    compressor->compress();
    compressor->saveToFile("compressedImg.frac");

    // Setup the decompressor with the compressed image
    Decompressor* decompressor = new RefDecompressor("compressedImg.frac");

    // setup the display
    const Image* img = decompressor->getImage();
    gDisplay.compressor = compressor;
    gDisplay.decompressor = decompressor;
    gDisplay.updateDecomp = false;
    gDisplay.width = img->width;
    gDisplay.height = img->height;

    // configure GLUT
    glutInitWindowSize(gDisplay.width, gDisplay.height);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("Fractal Image Compression");
    glutDisplayFunc(handleDisplay);
    glutKeyboardFunc(handleKeyPress);
    glutMainLoop();
}
