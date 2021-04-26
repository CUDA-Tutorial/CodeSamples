#include <cuda_runtime_api.h>
#include <iostream>


// Declaration of a device variable in constant memory
__constant__ int cFoo;

__global__ void ReadConstantMemory()
{
    printf("GPU: Reading constant memory --> %x\n", cFoo);
}

// Definition of a device variable in global memory
__device__ int dFoo = 42;

__global__ void ReadGlobalMemory(const int* __restrict dBar_ptr)
{
    printf("GPU: Reading global memory --> %d %x\n", dFoo, *dBar_ptr);
}

__global__ void WriteGlobalMemory(int* __restrict out_ptr)
{
    *out_ptr = dFoo * dFoo;
}

int main()
{
    // Uniform variables should best be placed in constant
    // GPU memory. Can be updated with cudaMemcpyToSymbol.
    int bar = 0xcaffe;
    cudaMemcpyToSymbol(cFoo, &bar, sizeof(int));
    ReadConstantMemory<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Larger or read-write data is easiest provisioned by
    // global memory. Can be allocated with cudaMalloc and
    // updated with cudaMemcpy. Must be freed afterward.
    int* dBarPtr;
    cudaMalloc((void**)&dBarPtr, sizeof(int));
    cudaMemcpy(dBarPtr, &bar, sizeof(int), cudaMemcpyHostToDevice);
    ReadGlobalMemory<<<1, 1>>>(dBarPtr);
    cudaDeviceSynchronize();
    cudaFree(dBarPtr);

    // The CPU may also read back updates from the GPU by
    // copying the relevant data from global memory after
    // running the kernel. Notice that here, we do not use
    // cudaDeviceSynchronize: cudaMemcpy will synchronize
    // with the CPU automatically.
    int out, *dOutPtr;
    cudaMalloc((void**)&dOutPtr, sizeof(int));
    WriteGlobalMemory<<<1,1>>>(dOutPtr);
    cudaMemcpy(&out, dOutPtr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dOutPtr);
    std::cout << "CPU: Copied back from GPU --> " << out << std::endl;

    return 0;
}
