#include <cuda_runtime_api.h>
#include <iostream>

__global__ void HelloGPU()
{
	// Print a simple message from the GPU
	printf("Hello from the GPU!\n");
}

int main()
{
	std::cout << "==== Sample 01 ====\n";
	std::cout << "==== Hello GPU ====\n" << std::endl;
	// Expected output: 12x "Hello from the GPU!\n"

	// Launch a kernel with 1 block that has 12 threads
	HelloGPU<<<1, 12>>>();

	/*
	 Synchronize with GPU to wait for printf to finish.
	 Results of printf are buffered and copied back to
	 the CPU for I/O after the kernel has finished.
	*/
	cudaDeviceSynchronize();
	return 0;
}