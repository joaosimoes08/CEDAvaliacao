#include "fft.h"
#include <cmath>
#include <omp.h>

// Bit-reversal permutation (DIT FFT)
void bit_reverse(ComplexVector& data) {
    int N = data.size();
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}

void fft_sequential(ComplexVector& data) {
    int N = data.size();
    if (N <= 1) return;

    bit_reverse(data); // Apply bit-reversal first

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

    bit_reverse(data); // Apply bit-reversal first

    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        double theta = -M_PI / step;

        #pragma omp parallel for num_threads(threads) schedule(guided)
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
