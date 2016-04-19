#include <string>

#include "cycleTimer.h"
#include "benchmark.h"

void CheckBenchmark(Compressor* ref_compressor, Compressor* cuda_compressor) {
    double startTime = CycleTimer::currentSeconds();
    ref_compressor->compress();
    double refTime = CycleTimer::currentSeconds() - startTime;

    startTime = CycleTimer::currentSeconds();
    cuda_compressor->compress();
    double cudaTime = CycleTimer::currentSeconds() - startTime;

    if (ref_compressor->getCompressedContents() == cuda_compressor->getCompressedContents()) {
        printf("Outputs Match!\n");
    } else {
        printf("Outputs Don't Match!\n");
    } 

    printf("Reference:   %.4f ms\n", 1000.f * refTime);
    printf("CUDA:    %.4f ms\n", 1000.f * cudaTime);
}
