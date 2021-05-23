#include <cuda_runtime_api.h>
#include <iostream>

__global__ void HelloGPU()
{
	// Print a simple message from the GPU
	printf("Hello from the GPU!\n");
}

int main()
{
	std::cout << "==== Sample 01 - Hello GPU ====\n" << std::endl;
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

/*
Exercises:
1) Change the message that is printed by the kernel
2) Write a different kernel (different name, different message)
3) Call the different kernels multiple times
*/