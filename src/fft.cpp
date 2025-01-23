#include "fft.h"
#include <cmath>
#include <omp.h>

void fft_sequential(CudaComplex* data, int N) {
    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        double theta = -M_PI / step;

        for (int group = 0; group < step; ++group) {
            CudaComplex w(cos(theta * group), sin(theta * group));
            for (int pair = group; pair < N; pair += jump) {
                CudaComplex temp = w * data[pair + step];
                data[pair + step] = data[pair] - temp;
                data[pair] += temp;
            }
        }
    }
}

void fft_parallel(CudaComplex* data, int N, int threads) {
    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        double theta = -M_PI / step;

        #pragma omp parallel for num_threads(threads)
        for (int group = 0; group < step; ++group) {
            CudaComplex w(cos(theta * group), sin(theta * group));
            for (int pair = group; pair < N; pair += jump) {
                CudaComplex temp = w * data[pair + step];
                data[pair + step] = data[pair] - temp;
                data[pair] += temp;
            }
        }
    }
}
