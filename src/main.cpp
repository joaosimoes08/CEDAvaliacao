#include <iostream>
#include "fft.h"
#include "timer.h"
#include <omp.h> // For OpenMP thread functions

int main() {
    int N;
    std::cout << "Enter the size of the data (must be a power of two): ";
    std::cin >> N;

    if ((N <= 0) || ((N & (N - 1)) != 0)) {
        std::cerr << "Error: Input size must be a power of two.\n";
        return 1;
    }

    ComplexVector data(N);

    // Initialize with example data
    for (int i = 0; i < N; ++i) {
        data[i] = Complex(i, 0);
    }

    // Sequential FFT
    ComplexVector seq_data = data;
    Timer t_seq;
    try {
        fft_sequential(seq_data);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    double time_seq = t_seq.elapsed();
    std::cout << "Sequential FFT: " << time_seq << "s\n";

    
    // Determine the number of threads dynamically
    int num_threads =  omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads for parallel FFT.\n";

    // Parallel FFT
    ComplexVector par_data = data;
    Timer t_par;
    try {
        fft_parallel(par_data, num_threads);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    double time_par = t_par.elapsed();
    std::cout << "Parallel Multicore FFT (" << num_threads << " threads): " << time_par << "s\n";

    // Calculate Speedup
    std::cout << "Speedup Multicore: " << time_seq / time_par << "\n";

    return 0;
}
