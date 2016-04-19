#ifndef __PPM_H__
#define __PPM_H__

struct Image;

void writePPMImage(const Image* image, const char *filename);
void readPPMImage(const char *filename, const Image* image);

#endif
