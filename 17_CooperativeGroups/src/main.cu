#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../../shared/include/utility.h"

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

    std::cout << "Producing random inputs..." << std::endl;
    // Generate some random numbers to reduce
    std::vector<float> vals;
    float* dValsPtr;
    samplesutil::prepareRandomNumbersCPUGPU(N, vals, &dValsPtr);
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
    std::cout << std::setw(20) << "Reduce Groups" << "\t" << ms << "ms \t" << mResult << "\n\n";
}

__managed__ unsigned int mHappyNumSum;
__managed__ unsigned int mHappyNumCount;

__global__ void happyNumbersGroups(unsigned int start, unsigned int N, unsigned int* mHappyNumbers)
{
    // Retrieve the input number based on the thread's global id
    unsigned int input = cg::this_grid().thread_rank() + start;
    // Compute whether or not the input number is in range and "happy" (utility function)
    bool happy = ((input-start) < N) && samplesutil::isHappy(input);
    // Create a group for the current warp
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    // Create a child group to separate threads with happy and unhappy numbers
    auto g = cg::binary_partition(warp, happy);

    if (happy)
    {
        // Compute the group's partial result of the sum of happy numbers
        unsigned int partial_sum = cg::reduce(g, input, cg::plus<unsigned int>());
        // One thread uses aggregate atomics to compute sum and write offset of happy numbers
        unsigned int offset;
        if (g.thread_rank() == 0)
        {
            atomicAdd(&mHappyNumSum, partial_sum);
            offset = atomicAdd(&mHappyNumCount, g.size());
        }
        // Distribute offset information from the thread that computed it to all others
        offset = g.shfl(offset, 0);
        // Each thread writes their happy number in a unique location
        mHappyNumbers[offset + g.thread_rank()] = input;
    }
}

void HappyNummbersWithGroups(unsigned int start, unsigned int N)
{
    // Initialize / allocate result storage. For brevity, we use managed memory
    mHappyNumSum = 0;
    mHappyNumCount = 0;
    unsigned int* mHappyNumbers;
    cudaMallocManaged((void**)&mHappyNumbers, sizeof(unsigned int) * N);

    // Compute count, sum and list of base 10 "happy numbers" from start to start+N
    happyNumbersGroups<<<(N + 255) / 256, 256>>>(start, N, mHappyNumbers);
    cudaDeviceSynchronize();

    // Print the count, sum and list of happy numbers in the given range
    std::cout << "No. of happy numbers in " << start << " - " << N << ": " << mHappyNumCount << "\n";
    std::cout << "Sum of happy numbers in " << start << " - " << N << ": " << mHappyNumSum << "\n";
    std::cout << "\nList of happy numbers in " << start << " - " << N << ": ";

    // Sort the managed memory happy number list in ascending order
    std::sort(mHappyNumbers, mHappyNumbers + mHappyNumCount);
    for (int i = 0; i < mHappyNumCount; i++)
        std::cout << mHappyNumbers[i] << ((i == mHappyNumCount - 1) ? "\n" : ", ");
}

int main()
{
    std::cout << "==== Sample 17 - Cooperative Groups ====\n" << std::endl;
    /*
    Cooperative groups are very versatile. They can be created for entire
    grids, blocks, warps or opportunistically for converged threads. In
    essence, they package a range of recent CUDA features in an interface
    that abstracts away the low-level instructions, making CUDA code 
    easier to understand. As such, cooperative groups have a vast range
    of applications. The two examples in this project cannot do them 
    justice, for further use cases please consider the advanced NVIDIA 
    CUDA Samples that include detailed, elaborate applications.

    Expected output:
    1) Result of reduction, now computed with cooperative groups
    2) The count, sum and list of the happy numbers in a given range (1-1000)
    */

    std::cout << "==== Computing a Reduction with Cooperative Groups ====" << std::endl;

    ReduceWithGroups();

    std::cout << "==== Computing Happy Numbers and their Sum ====" << std::endl;

    HappyNummbersWithGroups(1, 1000);
    
    return 0;
}