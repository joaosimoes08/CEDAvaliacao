#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
    }
};

#endif // TIMER_H
