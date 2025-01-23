#ifndef CUDA_COMPLEX_H
#define CUDA_COMPLEX_H

#include <cmath>

// Classe para números complexos compatível com CUDA
struct CudaComplex {
    double real;
    double imag;

    
    // Construtor padrão
    __host__ __device__ CudaComplex() : real(0.0), imag(0.0) {}

    // Construtor parametrizado
    __host__ __device__ CudaComplex(double r, double i) : real(r), imag(i) {}

    // Operador de multiplicação
    __host__ __device__ CudaComplex operator*(const CudaComplex& other) const {
        return CudaComplex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }

    // Operador de adição
    __host__ __device__ CudaComplex operator+(const CudaComplex& other) const {
        return CudaComplex(real + other.real, imag + other.imag);
    }

	
    // Operador de subtração
    __host__ __device__ CudaComplex operator-(const CudaComplex& other) const {
        return CudaComplex(real - other.real, imag - other.imag);
    }

    // Operador de atribuição com adição
    __host__ __device__ CudaComplex& operator+=(const CudaComplex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    // Operador de atribuição com subtração
    __host__ __device__ CudaComplex& operator-=(const CudaComplex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    // Operador de atribuição com multiplicação
    __host__ __device__ CudaComplex& operator*=(const CudaComplex& other) {
        double r = real * other.real - imag * other.imag;
        double i = real * other.imag + imag * other.real;
        real = r;
        imag = i;
        return *this;
    }
};
    
#endif // CUDA_COMPLEX_H
