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

int main()
{
    std::cout << "==== Sample 06 ====\n";
    std::cout << "==== Memory Basics ====\n" << std::endl;
    /*
     Expected output:
     "GPU: Reading constant memory-- > caffe"
     "GPU : Reading global memory-- > 42 caffe"
     "CPU : Copied back from GPU-- > 1764"
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

    return 0;
}
