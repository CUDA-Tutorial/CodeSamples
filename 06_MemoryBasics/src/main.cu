#include <cuda_runtime_api.h>
#include <iostream>


// Declaration of a device variable in constant memory
__constant__ int cFoo;

__global__ void ReadConstantMemory()
{
    printf("GPU: Reading constant memory --> %x\n", cFoo);
}

// Definition of a device variable in global memory
__device__ const int dFoo = 42;

__global__ void ReadGlobalMemory(const int* __restrict dBarPtr)
{
    printf("GPU: Reading global memory --> %d %x\n", dFoo, *dBarPtr);
}

__global__ void WriteGlobalMemory(int* __restrict dOutPtr)
{
    *dOutPtr = dFoo * dFoo;
}

__device__ void WriteAndPrintSharedMemory(int* sFoo)
{
    // Write a computed result to shared memory for other threads to see
    sFoo[threadIdx.x] = 42 * (threadIdx.x + 1);
    // We make sure that no thread prints while the other still writes (parallelism!)
    __syncwarp();
    // Print own computed result and result by neighbor
    printf("ThreadID: %d, sFoo[0]: %d, sFoo[1]: %d\n", threadIdx.x, sFoo[0], sFoo[1]);
}

__global__ void WriteAndPrintSharedMemoryFixed()
{
    // Fixed allocation of two integers in shared memory
    __shared__ int sFoo[2];
    // Use it for efficient exchange of information
    WriteAndPrintSharedMemory(sFoo);
}

__global__ void WriteAndPrintSharedMemoryDynamic()
{
    // Use dynamically allocated shared memory
    extern __shared__ int sFoo[];
    // Use it for efficient exchange of information
    WriteAndPrintSharedMemory(sFoo);
}

int main()
{
    std::cout << "==== Sample 06 - Memory Basics ====\n" << std::endl;
    /*
     Expected output:
        GPU: Reading constant memory --> caffe
        GPU: Reading global memory --> 42 caffe
        CPU: Copied back from GPU --> 1764

        Using static shared memory to share computed results
        ThreadID: 0, sFoo[0]: 42, sFoo[1]: 84
        ThreadID: 1, sFoo[0]: 42, sFoo[1]: 84

        Using dynamic shared memory to share computed results
        ThreadID: 0, sFoo[0]: 42, sFoo[1]: 84
        ThreadID: 1, sFoo[0]: 42, sFoo[1]: 84
    */

    const int bar = 0xcaffe;
    /*
     Uniform variables should best be placed in constant
     GPU memory. Can be updated with cudaMemcpyToSymbol.
     This syntax is unusual, but this is how it should be
    */
    cudaMemcpyToSymbol(cFoo, &bar, sizeof(int));
    ReadConstantMemory<<<1, 1>>>();
    cudaDeviceSynchronize();

    /*
     Larger or read-write data is easiest provisioned by
     global memory. Can be allocated with cudaMalloc and
     updated with cudaMemcpy. Must be free'd afterward.
    */
    int* dBarPtr;
    cudaMalloc((void**)&dBarPtr, sizeof(int));
    cudaMemcpy(dBarPtr, &bar, sizeof(int), cudaMemcpyHostToDevice);
    ReadGlobalMemory<<<1, 1>>>(dBarPtr);
    cudaDeviceSynchronize();
    cudaFree(dBarPtr);

    /*
     The CPU may also read back updates from the GPU by
     copying the relevant data from global memory after
     running the kernel. Notice that here, we do not use
     cudaDeviceSynchronize: cudaMemcpy will synchronize
     with the CPU automatically.
    */
    int out, *dOutPtr;
    cudaMalloc((void**)&dOutPtr, sizeof(int));
    WriteGlobalMemory<<<1,1>>>(dOutPtr);
    cudaMemcpy(&out, dOutPtr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dOutPtr);
    std::cout << "CPU: Copied back from GPU --> " << out << std::endl;

    /*
    For information that is shared only within a single threadblock,
    we can also use shared memory, which is usually more efficient than
    global memory. Shared memory for a block may be statically allocated
    inside the kernel, or dynamically allocated at the kernel launch. In
    the latter case, the size of the required shared memory is provided as
    the third launch parameter, and the kernel will be able to access the 
    allocated shared memory via an array with the "extern" decoration. 
    Below, we use both methods to provide shared memory for a kernel with 
    two threads that exchange computed integers. 
    */
    std::cout << "\nUsing static shared memory to share computed results" << std::endl;
    WriteAndPrintSharedMemoryFixed<<<1, 2>>>();
    cudaDeviceSynchronize();

    std::cout << "\nUsing dynamic shared memory to share computed results" << std::endl;
    WriteAndPrintSharedMemoryDynamic<<<1, 2, 2 * sizeof(int)>>>();
    cudaDeviceSynchronize();

    return 0;
}

/*
Exercises:
1) Write a function that takes data from constant memory and writes it to global. 
Copy it back from the GPU and print on the CPU.
2) Combine allocation, memcpy and several kernels in succession to produce a
more complex result. E.g., kernel A adds global values X + Y and writes the
result back to global, kernel B multiplies that result by Z. Convince yourself 
that the results remain in global memory between kernel launches and that a
kernel or a memcpy that runs after an earlier kernel can safely access the global
data that it produced, even if you don't use cudaDeviceSynchronize inbetween.
3) Try to write a kernel where one thread writes a value to shared memory without 
a syncwarp, so that other threads may fail to see it. You might need a block 
size larger than 32 threads for this to happen and you may have to let the writing 
thread do some "fake" work to delay its write to shared memory. Or it may work
immediately :) A solution should be provided by the following code sample.
*/