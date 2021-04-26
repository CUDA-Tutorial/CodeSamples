#include <cuda_runtime_api.h>
#include <iostream>

__global__ void HelloGPU()
{
	// Print a simple message from the GPU
	printf("Hello from the GPU!\n");
}

int main()
{
	// Launch a kernel with 1 block that has 12 threads
	HelloGPU<<<1, 12>>>();
	// Synchronize with GPU to wait for printf to finish.
	// Results of printf are buffered and copied back to
	// the CPU for I/O after the kernel has finished.
	cudaDeviceSynchronize();
	return 0;
}