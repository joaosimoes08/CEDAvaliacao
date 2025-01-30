#ifndef FFT_H
#define FFT_H

#include <complex>
#include <vector>

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// FFT functions
void fft_sequential(ComplexVector& data);
void fft_parallel(ComplexVector& data, int threads);
void fft_gpu(ComplexVector& data);

// Bit-reversal permutation
void bit_reverse(ComplexVector& data);

#endif // FFT_H
