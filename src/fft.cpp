#include "fft.h"
#include <cmath>
#include <omp.h>

// Bit-reversal permutation (paper Section I-B, Fig. 1)
// Reorders input data for DIT FFT to enable butterfly computation
void bit_reverse(ComplexVector& data) {
    int N = data.size();
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) std::swap(data[i], data[j]); // Swap mirrored indices
    }
}

int DecideThreadNum(int step) {
    int nMax_threadNum = step / MIN_WORK_PER_CORE;
    int tn = (nMax_threadNum > CORES_PER_NODE) ? CORES_PER_NODE : nMax_threadNum;
    return (tn < 1) ? 1 : tn;
}

// Sequential FFT (paper Section I-B, Cooley-Tukey DIT algorithm)
void fft_sequential(ComplexVector& data) {
    int N = data.size();
    if (N <= 1) return;

    bit_reverse(data); // Step 1: Permute input for DIT FFT

    // Step 2: Butterfly computation (log2(N) stages)
    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2; // Stride between butterfly pairs
        double theta = -M_PI / step; // Twiddle factor angle
		
        // Process each group of butterflies in this stage
        for (int group = 0; group < step; ++group) {
            Complex w = {std::cos(group * theta), std::sin(group * theta)}; // Twiddle factor
            // Apply butterfly to all pairs in this group
            for (int pair = group; pair < N; pair += jump) {
                Complex temp = w * data[pair + step];
                data[pair + step] = data[pair] - temp; // Upper wing
                data[pair] += temp;                     // Lower wing
            }
        }
    }
}

// OpenMP Parallel FFT (paper Section II-C, guided scheduling)
void fft_parallel(ComplexVector& data) {
    int N = data.size();
    if (N <= 1) return;

    bit_reverse(data); // Same as sequential

    for (int step = 1; step < N; step *= 2) {
    	int threads = DecideThreadNum(step);
        int jump = step * 2;
        double theta = -M_PI / step;

        // Parallelize butterfly groups across threads (paper Section II-B)
        #pragma omp parallel for num_threads(threads) schedule(guided) // Guided scheduling (paper Section II-C)
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
