#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <utility>
#include <numeric>

#define BLOCK_SIZE 256
#define WARMUP_ITERATIONS 10
#define TIMING_ITERATIONS 20
#define NUM_ITEMS 100'000'000

// Declare a GPU-visible floating point variable in global memory.
__device__ float dResult;

/*
 The most basic reduction kernel uses atomic operations to accumulate
 the individual inputs in a single, device-wide visible variable.
*/
__global__ void reduceAtomicGlobal(const float* input, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
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
__global__ void reduceAtomicShared(const float* input, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

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
__global__ void reduceShared(const float* input, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /*
    Use a larger shared memory region so that each
    thread can store its own partial results
    */
    __shared__ float data[BLOCK_SIZE];
    /*
    Use a new strategy to handle superfluous threads.
    To make sure they stay alive and can help with
    the reduction, threads without an input simply
    produce a '0', which as no effect on the result.
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
 Final improvement: using warp-level primitives to 
 accelerate the final steps of the reduction. Warps
 have a fast lane for communication. They are free 
 to exchange values in registers when they are being
 scheduled for execution. Warps will be formed from 
 consecutive threads in groups of 32.
*/
__global__ void reduceSharedShuffle(const float* input, int N)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

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

int main()
{
    std::cout << "==== Sample 07 ====\n";
    std::cout << "==== Reductions ====\n" << std::endl;
    /*
     Expected output: Accumulated results from CPU and GPU that 
     approach 42 * NUM_ITEMS (can vary greatly due to floating point 
     precision limitations). 
    
     With more sophisticated techniques, reported performance of the
     GPU versions (measured runtime in ms) should generally decrease.
    */

    const unsigned int N = NUM_ITEMS;
    const float target = 42.f;

    std::cout << "Producing random inputs...\n" << std::endl;
    // Generate a few random inputs to accumulate
    std::default_random_engine eng(0xcaffe);
    std::normal_distribution<float> dist(target);
    std::vector<float> vals(N);
    std::for_each(vals.begin(), vals.end(), [&](float& f) { f = dist(eng); });

    std::cout << "==== CPU Reduction ====\n" << std::endl;
    // A reference value is computed by sequential reduction
    std::cout << "Reference value: " << std::accumulate(vals.cbegin(), vals.cend(), 0.0f) << std::endl;
    // Print expected value, because reference may be off due to floating point (im-)precision
    std::cout << "\nExpected value: " << target * N << "\n" << std::endl;

    std::cout << "==== GPU Reductions ====\n" << std::endl;
    // Allocate some global GPU memory to write the inputs to
    float* dValsPtr;
    cudaMalloc((void**)&dValsPtr, sizeof(float) * N);
    // Expliclity copy the inputs from the CPU to the GPU
    cudaMemcpy(dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    // Compute the smallest grid to process N entries with a given block size
    dim3 blockDim = { BLOCK_SIZE, 1, 1 };
    dim3 gridDim = { (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };

    // Set up a collection of techniques to evaluate for performance
    std::pair<const char*, void(*)(const float*, int)> reductionTechniques[] = {
        std::make_pair("Atomic Global", reduceAtomicGlobal),
        std::make_pair("Atomic Shared", reduceAtomicShared),
        std::make_pair("Reduce Shared", reduceShared),
        std::make_pair("Reduce Shared + Shuffle", reduceSharedShuffle)
    };

    // Evaluate each technique
    for (const auto& [name, func] : reductionTechniques)
    {
        // Run several iterations for GPU warump
        for (int i = 0; i < WARMUP_ITERATIONS; i++)
            func<<<gridDim, blockDim>>>(dValsPtr, N);

        // Synchronize to ensure CPU only records time after warmup is done
        cudaDeviceSynchronize();
        auto before = std::chrono::system_clock::now();

        float result = 0.0f;
        // Run several iterations to get an average measurement
        for (int i = 0; i < TIMING_ITERATIONS; i++)
        {
            cudaMemcpyToSymbol(dResult, &result, sizeof(float));
            func<<<gridDim, blockDim>>>(dValsPtr, N);
        }

        // cudaMemcpyFromSymbol implicitly synchronizes CPU and GPU
        cudaMemcpyFromSymbol(&result, dResult, sizeof(float));
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - before).count();
        std::cout << elapsed / TIMING_ITERATIONS << "ms \t" << result << "\t" << name << std::endl;
    }

    // Free the allocated memory for input
    cudaFree(dValsPtr);
    return 0;
}
