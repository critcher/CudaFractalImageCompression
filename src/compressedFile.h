#ifndef __COMPRESSED_FILE_H__
#define __COMPRESSED_FILE_H__

#include <vector>

#include "fractal.h"

struct Image;

void readFracFile(const char *filename, CompressedImage* ci);
void writeFracFile(const CompressedImage& ci, const char *filename);

#endif
