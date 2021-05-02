// A simple kernel with two nested if clauses, 4 branches.
// Each thread will take a separate branch, and then perform
// N stepts. With legacy scheduling, each branch must be 
// finished before execution can continue with the next.
#include <random>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

namespace samplesutil
{
__host__ void prepareRandomNumbersCPUGPU(unsigned int N, std::vector<float>& vals, float** dValsPtr)
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
__device__ void takeNTurns(const char* who, unsigned int N)
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

__global__ void testScheduling(int N)
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

__host__ void run2NestedBranchesForNSteps(int N)
{
	testScheduling<<<1, 4>>>(N);
	cudaDeviceSynchronize();
}

__host__ __device__ bool isHappy(unsigned int num)
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