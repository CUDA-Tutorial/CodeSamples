#include <cuda_runtime_api.h>
#include <iostream>
#include "../../shared/include/utility.h"

__global__ void WriteSlow(int* out, int val)
{
    samplesutil::WasteTime(1'000'000'000ULL);
    // Finally write value
    *out = val;
}

__global__ void Square(int* out)
{
    *out = *out * *out;
}

__global__ void ApproximatePi(bool synchronized)
{
    // Create block-shared variable for approximated Pi
    __shared__ float sPi;
    // Thread 0 computes Pi and stores it to shared memory
    if (threadIdx.x == 0)
        sPi = samplesutil::GregoryLeibniz(100'000);

    // Boolean decides whether threads synchronize or not
    if (synchronized)
        __syncthreads();

    // Every thread should now perform some task with Pi
    if (threadIdx.x%32 == 0)
        printf("Thread %d thinks Pi = %f\n", threadIdx.x, sPi);
}

int main()
{
    std::cout << "==== Sample 07 - Synchronization ====\n" << std::endl;
    /*
    Expected output:
        Demonstrating implicit synchronization:
        42 squared = 1764

        No __syncthreads after computing a block-shared Pi:
        Thread 32 thinks Pi = 0.000000
        Thread 64 thinks Pi = 0.000000
        Thread 96 thinks Pi = 0.000000
        Thread 0 thinks Pi = 3.141586
        (or similar. Results may be correct, but not safe!)

        __syncthreads after computing a block-shared Pi:
        Thread 64 thinks Pi = 3.141586
        Thread 96 thinks Pi = 3.141586
        Thread 0 thinks Pi = 3.141586
        Thread 32 thinks Pi = 3.141586
        (or similar)
    */

    /*
    Implicit synchronization between kernels and cudaMemcpy:

    Consider the example below, where we have two kernels. The first
    kernel writes some data (slowly), the second modifies that data.
    Afterwards, we copy the modified data back to the CPU. By default,
    CUDA will assume that each command depends on the previous command
    and therefore will implicitly synchronize them: a kernel will only
    run when previous kernels have finished, note however that the CPU
    is free to continue working in the meantime. Similarly, cudaMemcpy 
    will only start when all previous kernels have finished, but it 
    will also make the CPU wait until the copy has finished. Hence, we 
    don't need any other synchronization in this scenario. 
    */
    std::cout << "Demonstrating implicit synchronization:" << std::endl;
    // Allocate some device memory for kernels to work with
    int* dFooPtr;
    cudaMalloc(&dFooPtr, sizeof(int));
    // First kernel sets device memory to 42 (slowly)
    WriteSlow<<<1,1>>>(dFooPtr, 42);
    // Second kernel squares value of variable
    Square<<<1,1>>>(dFooPtr);
    // Finally, we copy the result back to the CPU
    int foo;
    cudaMemcpy(&foo, dFooPtr, sizeof(int), cudaMemcpyDeviceToHost);
    // Print the result of the GPU's computation
    std::cout << "42 squared = " << foo << std::endl;

    /*
    Block-wide synchronization with syncthreads:

    The following kernels compute an approximation of Pi.
    The algorithm used is inherently sequential, therefore
    only one thread performs the communication and then 
    shares the result with all threads in the block. 
    However, while one thread is busy performing work, the
    other threads in the block are free to move ahead. 
    With __syncthreads, we force all threads in a block to 
    wait at a given point in the program until all other 
    threads get there.
    */
    std::cout << "\nNo __syncthreads after computing a block-shared Pi:" << std::endl;
    // Run once without syncthreads
    ApproximatePi<<<1, 128>>>(false);
    // Wait for printf to finish
    cudaDeviceSynchronize();

    std::cout << "\n__syncthreads after computing a block-shared Pi:" << std::endl;
    // Run again with syncthreads
    ApproximatePi<<<1, 128>>>(true);
    // Wait for printf to finish
    cudaDeviceSynchronize();

    return 0;
}

/*
Exercises:
1) Try launching a simple CUDA kernel 1000-10000 times in a loop, once
with cudaDeviceSynchronize after each launch, once without it. 
What's the effect on runtime? Does anything change about the program behavior?
2) You can also memcpy from device to device. Perform a few of them (e.g. moving a
value from device location A to device location B to C) and then back to CPU and
confirm that everything happened properly in order.
3) Try running a kernel where the first 16 threads in each warp take one branch,
the other 16 take the other, with a syncthreads in each branch. What happens?
Why? Document what happens when the first 32 threads in a block of size 64 take 
one branch, the other 32 the other, with a syncthreads in each branch. What happens 
now? Provide your best guess why.
*/