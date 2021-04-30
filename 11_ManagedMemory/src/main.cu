#include <cuda_runtime_api.h>
#include <iostream>

// Managed variables may be defined like device variables
__managed__ unsigned int mFoo;

// Print a managed variable
__global__ void PrintFoo()
{
	printf("mFoo GPU: %d\n", mFoo);
}

// Print a managed array of integers
__global__ void PrintBar(const int* mBarPtr, unsigned int numEntries)
{
	printf("mBar GPU: ");
	for (int i = 0; i < numEntries; i++)
		printf("%d%s", mBarPtr[i], (i == numEntries - 1) ? "\n" : ", ");
}

int main()
{
	std::cout << "==== Sample 11 - Managed Memory ====\n" << std::endl;

	/*
	Managed memory reduces code complexity by decoupling physical
	memory location from address range. The CUDA runtime will take
	care of moving the memory to the location where it is needed.
	No copies are required, but care must be taken for concurrent
	access. To avoid performance degradation, managed memory should
	be prefetched.
	
	Expected output: 
		mFoo GPU: 14
		mBar GPU: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
		mBar CPU: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13

		CUDA device does (NOT) support concurrent access
		mFoo GPU: 42
	*/

	constexpr unsigned int VALUE = 14;

	// We may assign values to managed variables on the CPU
	mFoo = VALUE;
	// Managed variables can be used without transferring
	PrintFoo<<<1,1>>>();
	// Wait for printf output
	cudaDeviceSynchronize();

	// We may also allocate managed memory on demand
	int* mBarPtr;
	cudaMallocManaged((void**)&mBarPtr, VALUE * sizeof(int));
	// Managed memory can be directly initialized on the CPU
	for (int i = 0; i < VALUE; i++)
		mBarPtr[i] = i;
	/*
	If we know ahead of time where managed memory will be used
	and performance is essential, we can prefetch it to the
	required location.
	*/
	int device;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(mBarPtr, VALUE * sizeof(int), device);
	// Launch kernel with managed memory pointer as parameter
	PrintBar<<<1,1>>>(mBarPtr, VALUE);
	// We may also prefetch it back to the CPU
	cudaMemPrefetchAsync(mBarPtr, VALUE * sizeof(int), cudaCpuDeviceId);
	// Wait for GPU printing and prefetching to finish
	cudaDeviceSynchronize();

	std::cout << "mBar CPU: ";
	for (int i = 0; i < VALUE; i++)
		std::cout << mBarPtr[i] << (i == VALUE - 1 ? "\n" : ", ");

	/*
	Devices may or may not support concurrent access to variables.
	If they don't, then the CPU must ensure that access to managed
	memory does not overlap with GPU kernel execution, even if the
	GPU does not use the managed memory in question. Support for
	concurrent access is queried via device properties. Modifying
	a variable on the CPU before a kernel is fine, because the kernel
	will only be launched if the CPU is done with prior instructions.
	*/
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	// Report support
	std::cout << "\nCUDA device does " << (!prop.concurrentManagedAccess ? "NOT " : "") << "support concurrent access\n";

	// Handling access to managed memory, depending on device properties
	mFoo = 42;
	PrintFoo<<<1, 1>>>();

	if (!prop.concurrentManagedAccess)
		// CPU access to managed memory and GPU execution may not overlap
		cudaDeviceSynchronize(); 
	
	// Modify on CPU after / during GPU execution
	mBarPtr[0] = 20;

	// Wait for results of printf
	cudaDeviceSynchronize();

	return 0;
}