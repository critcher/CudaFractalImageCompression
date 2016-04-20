#ifndef __REF_COMPRESSOR_H__
#define __REF_COMPRESSOR_H__

#include <vector>

#include "compressor.h"
#include "image.h"
#include "fractal.h"

class RefCompressor : public Compressor {

private:
    Image* image;
    std::string imageFilename;
    int rangeSize;
    int domainSize;
    std::vector<CodebookElement> codebook;
    std::vector<RangeBlockInfo> rangeBlockMapping;


    void generateCodebookEelements();
    CodebookElement generateIdentity(int x, int y, Image* fullImg);
    CodebookElement generateRotate90(int x, int y, Image* fullImg);
    CodebookElement generateRotate180(int x, int y, Image* fullImg);
    CodebookElement generateRotate270(int x, int y, Image* fullImg);
    CodebookElement generateFlip(int x, int y, Image* fullImg);
    CodebookElement generateFRot90(int x, int y, Image* fullImg);
    CodebookElement generateFRot180(int x, int y, Image* fullImg);
    CodebookElement generateFRot270(int x, int y, Image* fullImg);
    void getBestMapping();

public:
    RefCompressor(const std::string& imageFilename, int rangeSize, int domainSize);
    virtual ~RefCompressor();
    void compress();
    void saveToFile(const std::string& filename);
    std::string getCompressedContents();
};


#endif
