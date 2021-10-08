#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../../shared/include/utility.h"

__device__ int dVal = 42;
__device__ int dOut;

// Very simple kernel that updates a variable
__global__ void CopyVal(const int* val)
{
	// Simulating a little work
	samplesutil::WasteTime(1'000'000ULL);
	// Update a global value
	dOut = *val;
}

void checkForErrors()
{
	// Catch errors that can be detected without synchronization, clear them
	cudaError_t err;
	err = cudaGetLastError();
	if (err == cudaSuccess)
		std::cout << "cudaGetLastError() before sync found no error" << std::endl;
	else
		std::cout << "cudaGetLastError() before sync found error: " << cudaGetErrorName(err) << ", CLEARS ERROR" << std::endl;

	// Catch errors that require explicit synchronization, do not clear them
	err = cudaDeviceSynchronize();
	if (err == cudaSuccess)
		std::cout << "cudaDeviceSynchronize() found no error" << std::endl;
	else
		std::cout << "cudaDeviceSynchronize() found error: " << cudaGetErrorName(err) << ", KEEPS ERROR" << std::endl;

	// If errors were found via synchronization, cudaGetLastError clears them
	err = cudaGetLastError();
	if (err == cudaSuccess)
		std::cout << "cudaGetLastError() after sync found no error" << std::endl;
	else
		std::cout << "cudaGetLastError() after sync found error: " << cudaGetErrorName(err) << ", CLEARS ERROR" << std::endl;

	std::cout << std::endl;
}

#define PRINT_RUN_CHECK(S)		\
std::cout << #S << std::endl;	\
S;								\
checkForErrors();

int main()
{
	std::cout << "==== Sample 12 - Error Handling ====\n" << std::endl;
	/*
	 Many functions in the CUDA API return error codes that indicate
	 that something has gone wrong. However, this error is not 
	 necessarily caused by the function that returns it. Kernels and
	 asynchronous memcopies, e.g., return immediately and may only
	 encounter errors after the return value is observed on the CPU. 
	 Such errors can be detected at some later point, for instance by
	 a synchronous function like cudaMemcpy or cudaDeviceSynchronize,
	 or by cudaGetLastError after a synchronization. To ensure that 
	 every single CUDA call worked without error, we would have to 
	 sacrifice concurrency and asynchronicity. Hence, error checking 
	 is, in practice, rather opportunistic and happens e.g. at runtime 
	 when an algorithm is synchronized anyway or when we debug misbehaving 
	 code. The error checking in this code is thus not practical and only 
	 serves to illustrate how different mechanisms detect previous errors. 

	 Expected output:

		(CopyVal<<<1, 1>>>(validDAddress))
		cudaGetLastError() before sync found no error
		cudaDeviceSynchronize() found no error
		cudaGetLastError() after sync found no error

		(CopyVal<<<1, (1<<16)>>>(validDAddress))
		cudaGetLastError() before sync found error: cudaErrorInvalidConfiguration, CLEARS ERROR
		cudaDeviceSynchronize() found no error
		cudaGetLastError() after sync found no error

		(CopyVal<<<1, 1>>>(nullptr))
		cudaGetLastError() before sync found no error
		cudaDeviceSynchronize() found error: cudaErrorIllegalAddress, KEEPS ERROR
		cudaGetLastError() after sync found error: cudaErrorIllegalAddress, CLEARS ERROR

		cudaErrorInvalidPc: invalid program counter
	*/

	int* validDAddress;
	// A function may return an error code - should check those for success
	cudaError_t err = cudaGetSymbolAddress((void**)&validDAddress, dVal);

	if (err != cudaSuccess)
		// If an error occurred, identify it with cudaGetErrorName and react!
		std::cout << cudaGetErrorName(err) << std::endl;
	// Alternatively, you may peek at the last error to see if the program is ok
	err = cudaPeekAtLastError();
	// Getting the last error effectively resets it. Useful after reacting to it
	err = cudaGetLastError();

	/* 
	Launching a kernel with proper configuration and parameters.
	If the system is set up correctly, this should succeed.
	*/
	PRINT_RUN_CHECK((CopyVal<<<1, 1>>>(validDAddress)));

	/* 
	Launching a kernel with bigger block than possible.
	cudaGetLastError() can catch SOME errors without synchronizing!
	*/
	PRINT_RUN_CHECK((CopyVal<<<1, (1<<16)>>>(validDAddress)));

	/*
	Launching a kernel with invalid address - error occurs after launch.
	cudaGetLastError() alone may miss this without synchronization.
	*/
	PRINT_RUN_CHECK((CopyVal<<<1, 1>>>(nullptr)));

	// For any kind of error, CUDA also provides a more verbose description.
	std::cout << cudaGetErrorName(cudaErrorInvalidPc) << ": " << cudaGetErrorString(cudaErrorInvalidPc) << std::endl;
}

/*
Exercises:
1) Write a program that creates many pinned large allocations, and stop when 
the first error occurs. What is this error? When and why does it occur?
2) cudaMemcpy can implicitly synchronize the GPU and CPU, hence its return values
can be used to find any errors of kernels that were launched before it. Demonstrate
this for a simple example where a kernel does something illegal that you discover
using cudaMemcpy.
3) Try to produce an exotic error that does not occur already occur in this program
*/
