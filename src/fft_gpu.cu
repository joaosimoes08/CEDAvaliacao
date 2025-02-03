#include <cuda_runtime.h>
#include <cufft.h>
#include "fft.h"

// GPU kernel for bit-reversal permutation (paper Section I-B)
__global__ void bit_reverse_kernel(cufftComplex* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

	// Mirror index calculation (matches CPU implementation)
    int j = 0;
    int temp = i;
    for (int k = 1; k < N; k <<= 1) {
        j <<= 1;
        j |= temp & 1;
        temp >>= 1;
    }
    if (i < j) { // Swap mirrored indices
        cufftComplex temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

void fft_gpu(ComplexVector& data) {
    int N = data.size();
    cufftComplex *d_data, *h_data;

	// Copy data to GPU (paper Section III-B)
    h_data = new cufftComplex[N];
    for (int i = 0; i < N; i++) {
        h_data[i].x = data[i].real();
        h_data[i].y = data[i].imag();
    }

    cudaMalloc((void**)&d_data, N * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Execute cuFFT (paper Section III-A)
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Post-processing: Apply bit-reversal to match CPU output (paper Section III-C)
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bit_reverse_kernel<<<grid_size, block_size>>>(d_data, N);

	// Copy results back to CPU
    cudaMemcpy(h_data, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
        data[i] = Complex(h_data[i].x, h_data[i].y);
    }
    
	// Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    delete[] h_data;
}
