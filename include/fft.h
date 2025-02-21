#ifndef FFT_H
#define FFT_H

#include <complex>
#include <vector>

// Complex number and vector types (aligned with the paper's focus on DFT/FFT)
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// FFT functions (paper Section I & II)
void fft_sequential(ComplexVector& data);        // Baseline sequential FFT (Cooley-Tukey)
void fft_parallel(ComplexVector& data); // OpenMP parallel FFT (paper Section II)
void fft_gpu(ComplexVector& data);              // GPU-accelerated FFT (paper Section III)

// Bit-reversal permutation for Decimation-in-Time (DIT) FFT (paper Section I-B)
void bit_reverse(ComplexVector& data);

// Adaptive thread allocation
int DecideThreadNum(int step); // Add declaration

// Constants (define these based on your system)
constexpr int MIN_WORK_PER_CORE = 10;
constexpr int CORES_PER_NODE = 28;

#endif // FFT_H

