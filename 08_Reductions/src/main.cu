#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <utility>
#include <numeric>
#include <iomanip>
#include "../../shared/include/utility.h"

// Declare a GPU-visible floating point variable in global memory.
__device__ float dResult;

/*
 The most basic reduction kernel uses atomic operations to accumulate
 the individual inputs in a single, device-wide visible variable.
 If you have experience with atomics, it is important to note that the
 basic atomicXXX instructions of CUDA have RELAXED semantics (scary!).
 That means, the threads that operate atomically on them only agree that 
 there is a particular order for the accesses to that variable and nothing
 else (especially no acquire/release semantics).
*/
__global__ void reduceAtomicGlobal(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* 
    Since all blocks must have the same number of threads,
    we may have to launch more threads than there are 
    inputs. Superfluous threads should not try to read 
    from the input (out of bounds access!)
    */
    if (id < N)
        atomicAdd(&dResult, input[id]);
}

/*
 First improvement: shared memory is much faster than global
 memory. Each block can accumulate partial results in isolated
 block-wide visible memory. This relieves the contention on 
 a single global variable that all threads want access to.
*/
__global__ void reduceAtomicShared(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Declare a shared float for each block
    __shared__ float x;

    // Only one thread should initialize this shared value
    if (threadIdx.x == 0) 
        x = 0.0f;
    
    /*
    Before we continue, we must ensure that all threads
    can see this update (initialization) by thread 0
    */
    __syncthreads();

    /*
    Every thread in the block adds its input to the
    shared variable of the block.
    */
    if (id < N) 
        atomicAdd(&x, input[id]);

    // Wait until all threads have done their part
    __syncthreads();

    /*
    Once they are all done, only one thread must add
    the block's partial result to the global variable. 
    */
    if (threadIdx.x == 0) 
        atomicAdd(&dResult, x);
}

/*
 Second improvement: choosing a more suitable algorithm.
 We can exploit the fact that the GPU is massively parallel
 and come up with a fitting procedure that uses multiple
 iterations. In each iteration, threads accumulate partial 
 results from the previous iteration. Before, the contented
 accesses to one location forced the GPU to perform updates 
 sequentially O(N). Now, each thread can access its own, 
 exclusive shared variable in each iteration in parallel,
 giving an effective runtime that is closer to O(log N).
*/
template <unsigned int BLOCK_SIZE>
__global__ void reduceShared(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    /*
    Use a larger shared memory region so that each
    thread can store its own partial results
    */
    __shared__ float data[BLOCK_SIZE];
    /*
    Use a new strategy to handle superfluous threads.
    To make sure they stay alive and can help with
    the reduction, threads without an input simply
    produce a '0', which has no effect on the result.
    */
    data[threadIdx.x] = (id < N ? input[id] : 0);

    /*
    log N iterations to complete. In each step, a thread
    accumulates two partial values to form the input for
    the next iteration. The sum of all partial results 
    eventually yields the full result of the reduction. 
    */
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        /*
        In each iteration, we must make sure that all
        threads are done writing the updates of the
        previous iteration / the initialization.
        */
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }

    /*
    Note: thread 0 is the last thread to combine two
    partial results, and the one who writes to global
    memory, therefore no synchronization is required
    after the last iteration.
    */
    if (threadIdx.x == 0)
        atomicAdd(&dResult, data[0]);
}

/*
 Warp-level improvement: using warp-level primitives to 
 accelerate the final steps of the reduction. Warps
 have a fast lane for communication. They are free 
 to exchange values in registers when they are being
 scheduled for execution. Warps will be formed from 
 consecutive threads in groups of 32.
*/
template <unsigned int BLOCK_SIZE>
__global__ void reduceShuffle(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = (id < N ? input[id] : 0);

    // Only use shared memory until last 32 values
    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }

    // The last 32 values can be handled with warp-level primitives
    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        /* 
        The threads in the first warp shuffle their registers.
        This replaces the last 5 iterations of the previous solution.
        The mask indicates which threads participate in the shuffle.
        The value indicates which register should be shuffled. 
        The final parameter gives the source thread from which the
        current one should receive the shuffled value. Accesses that
        are out of range (>= 32) will wrap around, but are not needed
        (they will not affect the final result written by thread 0).
        In each shuffle, at least half of the threads only participate 
        so they can provide useful data from the previous shuffle for 
        lower threads. To keep the code short, we always let all threads
        participate, because it is an error to let threads reach a shuffle
        instruction that they don't participate in.
        */
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0)
        atomicAdd(&dResult, x);
}

/*
 Final improvement: half of our threads actually idle after 
 they have loaded data from global memory to shared! Better
 to have threads fetch two values at the start and then let
 them all do at least some meaningful work. This means that
 compared to all other methods, only half the number of 
 threads must be launched in the grid!
*/
template <unsigned int BLOCK_SIZE>
__global__ void reduceFinal(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];
    // Already combine two values upon load from global memory.
    data[threadIdx.x] = id < N ? input[id] : 0;
    data[threadIdx.x] += (id + N < 2*N) ? input[id + N] : 0;

    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }

    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0)
        atomicAdd(&dResult, x);
}

int main()
{
    std::cout << "==== Sample 08 - Reductions ====\n" << std::endl;
    /*
     Expected output: Accumulated results from CPU and GPU that 
     approach 42 * NUM_ITEMS (can vary greatly due to floating point 
     precision limitations). 
    
     With more sophisticated techniques, reported performance of the
     GPU versions (measured runtime in ms) should generally decrease.
    */

    constexpr unsigned int BLOCK_SIZE = 256;
    constexpr unsigned int WARMUP_ITERATIONS = 10;
    constexpr unsigned int TIMING_ITERATIONS = 20;
    constexpr unsigned int N = 100'000'000;

    std::cout << "Producing random inputs...\n" << std::endl;
    // Generate some random numbers to reduce
    std::vector<float> vals;
    float* dValsPtr;
    samplesutil::prepareRandomNumbersCPUGPU(N, vals, &dValsPtr);

    std::cout << "==== CPU Reduction ====\n" << std::endl;
    // A reference value is computed by sequential reduction
    std::cout << "Computed CPU value: " << std::accumulate(vals.cbegin(), vals.cend(), 0.0f) << std::endl;

    std::cout << "==== GPU Reductions ====\n" << std::endl;
    /*
     Set up a collection of reductions to evaluate for performance. 
     Each entry gives a technique's name, the kernel to call, and
     the number of threads required for each individual technique.
    */
    const std::tuple<const char*, void(*)(const float*, int), unsigned int> reductionTechniques[]
    {
        {"Atomic Global", reduceAtomicGlobal, N},
        {"Atomic Shared", reduceAtomicShared, N},
        {"Reduce Shared", reduceShared<BLOCK_SIZE>, N},
        {"Reduce Shuffle", reduceShuffle<BLOCK_SIZE>, N},
        {"Reduce Final", reduceFinal<BLOCK_SIZE>, N / 2 + 1}
    };

    // Evaluate each technique separately
    for (const auto& [name, func, numThreads] : reductionTechniques)
    {
        // Compute the smallest grid to start required threads with a given block size
        const dim3 blockDim = { BLOCK_SIZE, 1, 1 };
        const dim3 gridDim = { (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };

        // Run several reductions for GPU to warm up
        for (int i = 0; i < WARMUP_ITERATIONS; i++)
            func<<<gridDim, blockDim>>>(dValsPtr, N);

        // Synchronize to ensure CPU only records time after warmup is done
        cudaDeviceSynchronize();
        const auto before = std::chrono::system_clock::now();

        float result = 0.0f;
        // Run several iterations to get an average measurement
        for (int i = 0; i < TIMING_ITERATIONS; i++)
        {
            // Reset acummulated result to 0 in each run
            cudaMemcpyToSymbol(dResult, &result, sizeof(float));
            func<<<gridDim, blockDim>>>(dValsPtr, N);
        }

        // cudaMemcpyFromSymbol will implicitly synchronize CPU and GPU
        cudaMemcpyFromSymbol(&result, dResult, sizeof(float));

        // Can measure time without an extra synchronization
        const auto after = std::chrono::system_clock::now();
        const auto elapsed = 1000.f * std::chrono::duration_cast<std::chrono::duration<float>>(after - before).count();
        std::cout << std::setw(20) << name << "\t" << elapsed / TIMING_ITERATIONS << "ms \t" << result << std::endl;
    }

    // Free the allocated memory for input
    cudaFree(dValsPtr);
    return 0;
}

/*
Exercises: 
1) Change the program so that the methods reduce integer values instead of float. 
Can you observe any difference in terms of speed / computed results?
2) Do you have any other ideas how the reduction could be improved?
Making it even faster should be quite challenging, but if you have 
some suggestions, try them out and see how they affect performance! 
*/
