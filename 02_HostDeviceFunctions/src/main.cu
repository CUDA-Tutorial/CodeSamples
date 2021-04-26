#include <cuda_runtime_api.h>
#include <iostream>

// Define a function that will only be compiled for and called from host
__host__ void HostOnly()
{
    std::cout << "This function may only be called from the host" << std::endl;
}

// Define a function that will only be compiled for and called from device
__device__ void DeviceOnly()
{
    printf("This function may only be called from the device\n");
}

// Define a function that will be compiled for both architectures
__host__ __device__ float SquareAnywhere(float x)
{
    return x * x;
}

// Call device and portable functions from a kernel
__global__ void RunGPU(float x)
{
    DeviceOnly();
    printf("%f\n", SquareAnywhere(x));
}

// Call host and portable functions from a kernel
// Note that, by default, if function has no architecture
// specified, it is assumed to be __host__ by NVCC.
void RunCPU(float x)
{
    HostOnly();
    std::cout << SquareAnywhere(x) << std::endl;
}

int main()
{
    RunCPU(42);
    RunGPU<<<1, 1>>>(42);
    cudaDeviceSynchronize();
    return 0;
}
