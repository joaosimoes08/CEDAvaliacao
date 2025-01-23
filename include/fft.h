#ifndef FFT_H
#define FFT_H

#include "cuda_complex.h"

void fft_sequential(CudaComplex* data, int N);
void fft_parallel(CudaComplex* data, int N, int threads);
void fft_gpu(CudaComplex* data, int N);

#endif // FFT_H
