#include <iostream>
#include "fft.h"
#include "timer.h"

int main() {
    const int N = 1099511627776; // Exemplo com 1024 elementos
    ComplexVector data(N);

    // Inicializar com dados de exemplo
    for (int i = 0; i < N; ++i) {
        data[i] = Complex(i, 0);
    }

    // Sequencial
    ComplexVector seq_data = data;
    Timer t_seq;
    fft_sequential(seq_data);
    double time_seq = t_seq.elapsed();
    std::cout << "Sequencial: " << time_seq << "s\n";

    // Paralela Multicore
    ComplexVector par_data = data;
    Timer t_par;
    fft_parallel(par_data, 4); // 4 threads
    double time_par = t_par.elapsed();
	std::cout << "Paralela Multicore: " << time_par << "s\n";

    ComplexVector gpu_data = data;
    Timer t_gpu;
    fft_gpu(gpu_data);
    double time_gpu = t_gpu.elapsed();
    std::cout << "GPU: " << time_gpu << "s\n";


    std::cout << "Speedup Multicore: " << time_seq / time_par << "\n";
    std::cout << "Speedup GPU: " << time_seq / time_gpu << "\n";
    std::cout << "Speed GPU vs MultiCore: " << time_par / time_gpu << "\n";

    return 0;
}
