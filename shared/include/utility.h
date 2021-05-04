// A simple kernel with two nested if clauses, 4 branches.
// Each thread will take a separate branch, and then perform
// N stepts. With legacy scheduling, each branch must be 
// finished before execution can continue with the next.
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <mma.h>

#ifndef SAMPLES_UTIL_INCLUDED
#define SAMPLES_UTIL_INCLUDED

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
template <typename F>
struct Matrix {
    int width;
    int height;
    int stride;
    F* elements;
};

namespace samplesutil
{
    // Helper function to let threads spin
    __device__ void WasteTime(unsigned long long duration)
    {
        const unsigned long long int start = clock64();
        while ((clock64() - start) < duration);
    }

    __device__ float GregoryLeibniz(unsigned int iterations)
    {
        float pi = 0.f, m = 1.f;
        for (int n = 0; n < iterations; n++, m *= -1.f)
            pi += 4.f * (m / (2 * n + 1));
        return pi;
    }

    // Get a matrix element
    template <typename F>
    __device__ float GetElement(const F* A, int row, int col, unsigned int DIM)
    {
        return A[row * DIM + col];
    }

    // Set a matrix element
    template <typename F>
    __device__ void SetElement(F* A, int row, int col, float value, unsigned int DIM)
    {
        A[row * DIM + col] = value;
    }

    // Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
    // located col sub-matrices to the right and row sub-matrices down
    // from the upper-left corner of A
    template <typename F>
    __device__ F* GetSubMatrix(F* A, int row, int col, unsigned int BLOCK_SIZE, unsigned int DIM)
    {
        F* Asub = A + DIM * BLOCK_SIZE * row + BLOCK_SIZE * col;
        return Asub;
    }

    // Matrix multiplication kernel called by MatMul()
    template <unsigned int DIM, unsigned int BLOCK_SIZE>
    __global__ void MatMulKernel(half* A, half* B, float* C)
    {
        // Block row and column
        int blockRow = blockIdx.y;
        int blockCol = blockIdx.x;

        // Each thread block computes one sub-matrix Csub of C
        float* Csub = GetSubMatrix(C, blockRow, blockCol, BLOCK_SIZE, DIM);

        // Each thread computes one element of Csub
        // by accumulating results into Cvalue
        float Cvalue = 0;

        // Thread row and column within Csub
        int row = threadIdx.y;
        int col = threadIdx.x;

        // Loop over all the sub-matrices of A and B that are
        // required to compute Csub
        // Multiply each pair of sub-matrices together
        // and accumulate the results
        for (int m = 0; m < (DIM / BLOCK_SIZE); ++m) {

            // Get sub-matrix Asub of A
            half* Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE, DIM);

            // Get sub-matrix Bsub of B
            half* Bsub = GetSubMatrix(B, m, blockCol, BLOCK_SIZE, DIM);

            // Shared memory used to store Asub and Bsub respectively
            __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load Asub and Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            As[row][col] = GetElement(Asub, row, col, DIM);
            Bs[row][col] = GetElement(Bsub, row, col, DIM);

            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            __syncthreads();
            // Multiply Asub and Bsub together
            for (int e = 0; e < BLOCK_SIZE; ++e)
                Cvalue += (float)As[row][e] * (float)Bs[e][col];

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write Csub to device memory
        // Each thread writes one element
        SetElement(Csub, row, col, Cvalue, DIM);
    }

    // Matrix multiplication - Host code
    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    template <typename K>
    static void MatMul(const Matrix<half> A, const Matrix<half> B, Matrix<float> C, const K& kernel, dim3 grid, dim3 block)
    {
        // Load A and B to device memory
        Matrix<half> d_A;
        d_A.width = d_A.stride = A.width; d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(half);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size,
            cudaMemcpyHostToDevice);
        Matrix<half> d_B;
        d_B.width = d_B.stride = B.width; d_B.height = B.height;
        size = B.width * B.height * sizeof(half);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size,
            cudaMemcpyHostToDevice);

        // Allocate C in device memory
        Matrix<float> d_C;
        d_C.width = d_C.stride = C.width; d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // Invoke kernel
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_A, d_B, d_C);
        cudaEventRecord(end);

        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        std::cout << "Matrix multiplication took: " << ms << " ms\n" << std::endl;

        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size,
            cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }

    template <unsigned int BLOCK_SIZE>
    static void MatMulConv(Matrix<half> A, Matrix<half> B, Matrix<float> C)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(B.width / block.x, A.height / block.y);
        samplesutil::MatMul(A, B, C, samplesutil::MatMulKernel<BLOCK_SIZE>, grid, block);
    }

    static __host__ void prepareRandomNumbersCPUGPU(unsigned int N, std::vector<float>& vals, float** dValsPtr)
    {
	    constexpr float target = 42.f;
	    // Print expected value, because reference may be off due to floating point (im-)precision
	    std::cout << "\nExpected value: " << target * N << "\n" << std::endl;

	    // Generate a few random inputs to accumulate
	    std::default_random_engine eng(0xcaffe);
	    std::normal_distribution<float> dist(target);
	    vals.resize(N);
	    std::for_each(vals.begin(), vals.end(), [&dist, &eng](float& f) { f = dist(eng); });

	    // Allocate some global GPU memory to write the inputs to
	    cudaMalloc((void**)dValsPtr, sizeof(float) * N);
	    // Expliclity copy the inputs from the CPU to the GPU
	    cudaMemcpy(*dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    }

    // Define an unsigned integer variable that the GPU can work with
    __device__ unsigned int step = 0;

    // Increment the GPU variable N times. Whenever a thread observes
    // non-consecutive numbers, it prints the latest sequence. Hence,
    // every thread documents the turns that it was given by the 
    // scheduler. 
    static __device__ void takeNTurns(const char* who, unsigned int N)
    {
        unsigned int lastTurn = 0, turn, start;
        for (int i = 0; i < N; i++)
        {
            turn = atomicInc(&step, 0xFFFFFFFFU);
            if (lastTurn != (turn-1))
                start = turn;

            if ((i == N - 1) || ((i > 0) && (start == turn)))
                printf("%s: %d--%d\n", who, start, turn);

            lastTurn = turn;
        }
    }

    static __global__ void testScheduling(int N)
    {
        if (threadIdx.x < 2) // Branch once
            if (threadIdx.x == 0) // Branch again
                takeNTurns("Thread 1", N);
            else
                takeNTurns("Thread 2", N);
        else
            if (threadIdx.x == 2) // Branch again
                takeNTurns("Thread 3", N);
            else
                takeNTurns("Thread 4", N);
    }

    static __host__ void run2NestedBranchesForNSteps(int N)
    {
	    testScheduling<<<1, 4>>>(N);
	    cudaDeviceSynchronize();
    }

    /*
     Computes whether a given number is a "happy number".
     https://en.wikipedia.org/wiki/Happy_number
    */
    static __host__ __device__ bool isHappy(unsigned int num)
    {
        while (num != 0 && num != 1 && num != 4)
        {
            unsigned int next_num = 0;
            for (unsigned int n = num; n > 0; n /= 10)
            {
                unsigned int t = n % 10;
                next_num += t * t;
            }
            num = next_num;
        }
        return num == 1;
    }
}

#endif