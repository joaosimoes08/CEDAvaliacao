#include <iostream>
#include <fstream>
#include "fft.h"
#include "timer.h"

// Adaptive thread allocation from paper Section II-C
int DecideThreadNum(int nRound, int nMinRoundPerCore, int maxCores) {
    int nMax_threadNum = nRound / nMinRoundPerCore;
    int tn = (nMax_threadNum > maxCores) ? maxCores : nMax_threadNum;
    return (tn < 1) ? 1 : tn;
}
// Test problem sizes from 2^10 to 2^26 (paper Section III-D)
int main() {
    const int MIN_WORK_PER_CORE = 10;  
    const int CORES_PER_NODE = 28;      // 28 cores per AMD Epyc node
    // Open a file to save the results
    std::ofstream results("speedup_results.csv");
    results << "N,Sequential Time (s),Multicore Time (s),GPU Time (s),Threads Used,Speedup Multicore,Speedup GPU,Speedup GPU vs Multicore\n";

    // Loop through powers of 2 from 2^10 to 2^26
    for (int exp = 10; exp <= 26; ++exp) {
        int N = 1 << exp; // 2^exp
        ComplexVector data(N);

        // Initialize with a cosine wave (paper Section III-D)
        for (int i = 0; i < N; ++i) {
            data[i] = Complex(cos(2 * M_PI * i / N), 0); // Example: cosine wave
        }

        // Measure sequential FFT
        ComplexVector seq_data = data;
        Timer t_seq;
        fft_sequential(seq_data);
        double time_seq = t_seq.elapsed();

        // nRound = Total butterfly groups in the last stage (N/2)
        int nRound = N / 2;  // Fix: Use actual parallel workload size
        int threads = DecideThreadNum(nRound, MIN_WORK_PER_CORE, CORES_PER_NODE);

        // Parallel FFT with adaptive threads
        ComplexVector par_data = data;
        Timer t_par;
        fft_parallel(par_data, threads);
        double time_par = t_par.elapsed();

        // Measure GPU FFT (paper Section III)
        ComplexVector gpu_data = data;
        Timer t_gpu;
        fft_gpu(gpu_data);
        double time_gpu = t_gpu.elapsed();

        // Compute speedups (paper Section III-D)
        double speedup_multicore = time_seq / time_par;
        double speedup_gpu = time_seq / time_gpu;
        double speedup_gpu_vs_multicore = time_par / time_gpu;

        // Save results for analysis
        results << N << "," << time_seq << "," << time_par << "," << time_gpu << ","
                << threads << "," << speedup_multicore << "," << speedup_gpu << ","
                << speedup_gpu_vs_multicore << "\n";

        std::cout << "N = 2^" << exp << " (" << N << "): Used " << threads << " threads\n";
    }

    results.close();
    std::cout << "Results saved to speedup_results.csv\n";
    return 0;
}
