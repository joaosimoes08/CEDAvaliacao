#include <cuda_runtime.h>
#include <cufft.h>
#include "fft.h"

void fft_gpu(ComplexVector& data) {
    int N = data.size();
    cufftComplex *d_data, *h_data;
    
    // Allocate host memory
    h_data = new cufftComplex[N];
    
    // Convert input data
    for (int i = 0; i < N; i++) {
        h_data[i].x = data[i].real();
        h_data[i].y = data[i].imag();
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_data, N * sizeof(cufftComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // Create and execute FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    // Copy result back
    cudaMemcpy(h_data, d_data, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    // Convert output data
    for (int i = 0; i < N; i++) {
        data[i] = Complex(h_data[i].x, h_data[i].y);
    }
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    delete[] h_data;
}
