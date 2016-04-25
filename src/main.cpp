#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "refCompressor.h"
#include "cudaCompressor.h"
#include "display.h"
#include "benchmark.h"
#include "platformgl.h"
#include "ppm.h"

void usage(const char* progname) {
    printf("Usage: %s [options] imageFilename\n", progname);
    printf("Program Options:\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -r  --renderer <ref/cuda>  Select renderer: ref or cuda\n");
    printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{
    std::string imageFilename;
    bool useRefCompressor = true;

    bool checkCorrectness = false;
    int rangeSize = 16;
    int domainSize = 32;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  '?'},
        {"accuracy",    0, 0,  'a'},
        {"compressor", 1, 0,  'c'},
        {"domainSize", 1, 0,  'd'},
        {"rangeSize", 1, 0,  'r'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "c:d:r:a?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'a':
            checkCorrectness = true;
            break;
        case 'c':
            if (std::string(optarg).compare("cuda") == 0) {
                useRefCompressor = false;
            }
            break;
        case 'd':
            domainSize = atoi(optarg);
            break;
        case 'r':
            rangeSize = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////


    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing image filename\n");
        usage(argv[0]);
        return 1;
    }

    imageFilename = argv[optind];

    Compressor* compressor;

    if (checkCorrectness) {
        Compressor* cudaCompressor;

        compressor = new RefCompressor(imageFilename, rangeSize, domainSize);
        //cudaCompressor = new CudaCompressor(imageFilename, 1, 1);

        // Check the correctness
        CheckBenchmark(compressor, cudaCompressor);
        delete compressor;
        delete cudaCompressor;
    }
    else {
        if (useRefCompressor)
            compressor = new RefCompressor(imageFilename, rangeSize, domainSize);
        else {
            //compressor = new CudaCompressor(imageFilename, 1, 1);
        }

        glutInit(&argc, argv);
        startCompressionWithDisplay(compressor);
        delete compressor;
    }

    return 0;
}
