#include "cuda_complex.h"
#include <cuda_runtime.h>

// Kernel CUDA
__global__ void fft_kernel(CudaComplex* data, int N, int step, double theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N / (2 * step)) return;

    int pair = idx * 2 * step;
    CudaComplex w(cos(theta * idx), sin(theta * idx));
    CudaComplex temp = w * data[pair + step];
    data[pair + step] = data[pair] - temp;
    data[pair] += temp;
}

// Implementação FFT usando CUDA
void fft_gpu(CudaComplex* data, int N) {
    if (N <= 1) return;

    CudaComplex* d_data;
    cudaMalloc(&d_data, N * sizeof(CudaComplex));
    cudaMemcpy(d_data, data, N * sizeof(CudaComplex), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    for (int step = 1; step < N; step *= 2) {
        double theta = -M_PI / step;
        int numBlocks = (N / (2 * step) + threadsPerBlock - 1) / threadsPerBlock;
        fft_kernel<<<numBlocks, threadsPerBlock>>>(d_data, N, step, theta);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data, d_data, N * sizeof(CudaComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
