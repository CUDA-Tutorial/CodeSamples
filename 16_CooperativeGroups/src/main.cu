#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../../shared/include/generate_random.h"

// Shortening cooperative groups namespace for convenience
namespace cg = cooperative_groups;

// We keep the result of the reduction in managed memory
__managed__ float mResult;

template <unsigned int BLOCK_SIZE>
__global__ void reduceGroups(const float* __restrict input, int N)
{
    // Can conveniently obtain groups for grid and block
    auto block = cg::this_thread_block();
    unsigned int gId = cg::this_grid().thread_rank();

    __shared__ float data[BLOCK_SIZE];
    data[block.thread_rank()] = (gId < N ? (input[gId] + input[gId + N / 2]) : 0);

    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        // Rather than selecting explicit sync functions, groups offer sync()
        block.sync();
        if (block.thread_rank() < s)
            data[block.thread_rank()] += data[block.thread_rank() + s];
    }

    // Splitting blocks into warp groups is cleaner than checking threadIdx
    auto warp = cg::tiled_partition<32>(block);
    if (warp.meta_group_rank() == 0)
    {
        // Reduction primitives - will be hardware-accelerated on CC 8.0+
        float v = cg::reduce(warp, data[warp.thread_rank()], cg::plus<float>());
        if (warp.thread_rank() == 0)
            atomicAdd(&mResult, v);
    }
}

void ReduceWithGroups()
{
    constexpr unsigned int BLOCK_SIZE = 256, N = 1'000'000;

    std::cout << "Producing random inputs...\n" << std::endl;
    // Generate some random numbers to reduce
    std::vector<float> vals;
    float* dValsPtr;
    prepareRandomNumbersCPUGPU(N, vals, &dValsPtr);
    // Prepare grid configuration for input and used reduction technique
    const dim3 blockDim = { BLOCK_SIZE, 1, 1 };
    const dim3 gridDim = { (N / 2 + BLOCK_SIZE) / BLOCK_SIZE, 1, 1 };

    // Events for measuring run time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Setting managed result variable
    mResult = 0;
    cudaEventRecord(start);
    reduceGroups<BLOCK_SIZE><<<gridDim, blockDim>>>(dValsPtr, N);
    cudaEventRecord(end);

    float ms;
    // Synchronizing to event. Event is last, same effect as cudaDeviceSynchronize
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    std::cout << std::setw(20) << "Reduce Groups" << "\t" << ms << "ms \t" << mResult << std::endl;
}

__managed__ unsigned int mHappyNumSum;
__managed__ unsigned int mHappyNumCount;

__device__ bool isHappy(unsigned int num)
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

__global__ void Sum10HappyNumbers(unsigned int N, unsigned int* mHappyNumbers)
{
    unsigned int input = cg::this_grid().thread_rank() + 1;

    bool happy = (input <= N) && isHappy(input);

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto g = cg::binary_partition(warp, happy);

    if (happy)
    {
        unsigned int offset;
        unsigned int partial_sum = cg::reduce(g, input, cg::plus<unsigned int>());
        if (g.thread_rank() == 0)
        {
            atomicAdd(&mHappyNumSum, partial_sum);
            offset = atomicAdd(&mHappyNumCount, g.size());
        }
        offset = g.shfl(offset, 0);
        mHappyNumbers[offset + g.thread_rank()] = input;
    }
}

void HappyNummbersWithGroups(unsigned int N)
{
    mHappyNumSum = 0;
    mHappyNumCount = 0;
    unsigned int* mHappyNumbers;
    cudaMallocManaged((void**)&mHappyNumbers, sizeof(unsigned int) * N);

    Sum10HappyNumbers<<<(N + 255) / 256, 256>>>(N, mHappyNumbers);
    cudaDeviceSynchronize();

    std::cout << "No. of happy numbers from 1 - " << N << ": " << mHappyNumCount << std::endl;
    std::cout << "Sum of happy numbers from 1 - " << N << ": " << mHappyNumSum << std::endl;
    std::cout << "\nList of happy numbers from 1 - " << N << ": ";

    std::sort(mHappyNumbers, mHappyNumbers + mHappyNumCount);
    for (int i = 0; i < mHappyNumCount; i++)
        std::cout << mHappyNumbers[i] << ((i == mHappyNumCount - 1) ? "\n" : ", ");
}

int main()
{
    std::cout << "==== Sample 16 - Cooperative Groups ====\n" << std::endl;
    /*
    Cooperative groups are very versatile. They can be created for entire
    grids, blocks, warps or opportunistically for converged threads. In
    essence, they package a range of recent CUDA features in an interface
    that abstracts away the low-level instructions, making CUDA code 
    easier to understand. As such, cooperative groups have a vast range
    of applications. The examples in this project cannot do them justice,
    for further use cases please consider the advanced NVIDIA Samples 
    that include detailed, elaborate applications.

    Expected output:
    1) Result of reduction, now computed with cooperative groups
    */

    //ReduceWithGroups();

    //RejectionSamplePiWithGroups();
    HappyNummbersWithGroups(1000);
    
    return 0;
}