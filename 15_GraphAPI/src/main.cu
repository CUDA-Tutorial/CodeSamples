#include <cuda_runtime_api.h>
#include <iostream>

int main()
{
	std::cout << "==== Sample 13 - Graph API ====\n" << std::endl;

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


	return 0;
}