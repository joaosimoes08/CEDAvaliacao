#include "fft.h"
#include <cmath>
#include <omp.h>

void fft_sequential(ComplexVector& data) {
    int N = data.size();
    if (N <= 1) return;

    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        double theta = -M_PI / step;

        for (int group = 0; group < step; ++group) {
            Complex w = {std::cos(group * theta), std::sin(group * theta)};
            for (int pair = group; pair < N; pair += jump) {
                Complex temp = w * data[pair + step];
                data[pair + step] = data[pair] - temp;
                data[pair] += temp;
            }
        }
    }
}

void fft_parallel(ComplexVector& data, int threads) {
    int N = data.size();
    if (N <= 1) return;

    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        double theta = -M_PI / step;

        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int group = 0; group < step; ++group) {
            Complex w = {std::cos(group * theta), std::sin(group * theta)};
            for (int pair = group; pair < N; pair += jump) {
                Complex temp = w * data[pair + step];
                data[pair + step] = data[pair] - temp;
                data[pair] += temp;
            }
        }
    }
}
