#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

#endif // TIMER_H
