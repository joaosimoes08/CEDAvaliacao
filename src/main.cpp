#include <iostream>
#include "fft.h"
#include "timer.h"

int main() {
    const int N = 8; // Exemplo com 8 elementos
    CudaComplex data[N];

    for (int i = 0; i < N; ++i) {
        data[i] = CudaComplex(i, 0);
    }

    // FFT Sequencial
    Timer t1;
    fft_sequential(data, N);
    std::cout << "FFT Sequencial demorou " << t1.elapsed() << " segundos.\n";

    // FFT Paralela com OpenMP
    Timer t2;
    fft_parallel(data, N, 4);
    std::cout << "FFT Paralela demorou " << t2.elapsed() << " segundos.\n";

    // FFT com GPU
    Timer t3;
    fft_gpu(data, N);
    std::cout << "FFT GPU demorou " << t3.elapsed() << " segundos.\n";

    return 0;
}
