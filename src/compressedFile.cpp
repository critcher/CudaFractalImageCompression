#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "image.h"
#include "compressedFile.h"

void readFracFile(const char *filename, CompressedImage* ci) {
    std::ifstream f(filename);

    f >> ci->width >> ci->height >> ci->rangeSize >> ci->domainSize;

    while(f) {
        int trans;
        RangeBlockInfo r;
        r.codebookElement = new CodebookElement();
        f >> r.codebookElement->x >> r.codebookElement->y >> trans >> r.brightnessOffset >> r.contrastFactor;
        r.codebookElement->transform = static_cast<Transform>(trans);
        ci->rangeInfo.push_back(r);
    }
}

void writeFracFile(const CompressedImage& ci, const char *filename) {
    std::ofstream f(filename);

    std::string sep = " ";

    f << ci.width << sep << ci.height << "\n";
    f << ci.rangeSize << sep << ci.domainSize << "\n";

    for (unsigned int i = 0; i < ci.rangeInfo.size(); i++) {
        RangeBlockInfo r = ci.rangeInfo[i];
        f << r.codebookElement->x << sep << r.codebookElement->y << sep << r.codebookElement->transform << sep << r.brightnessOffset << sep << r.contrastFactor << "\n";
    }
}
