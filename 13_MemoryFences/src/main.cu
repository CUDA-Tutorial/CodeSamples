#include <cuda_runtime_api.h>
#include <iostream>
#include "../../shared/include/utility.h"

/*
 Producer function.

 Following a threadfence (memory barrier) with a volatile yields a release pattern.
 Following a threadfence (memory barrier) with an atomic yields a release pattern.
 Note however, that neither of these options is ideal. For one, we combine two
 operations to achieve a certain behavior. Second, threadfence is a general memory
 barrier, and is thus stronger than it may have to be (e.g., release barrier only).
 Volta+ actually have support for memory coherency models with proper acquire /
 release semantics, which are exposed to the programmer via cuda::std::atomic in
 libcu++.
*/
template <bool ATOMIC>
__device__ void ProduceFoo(unsigned int id, float* dFooPtr, int *dFooReadyPtr)
{
	float pi = samplesutil::GregoryLeibniz(10'000'000);
	dFooPtr[id] = pi;

	__threadfence();

	if (ATOMIC)
		atomicExch(&dFooReadyPtr[id], 1);
	else
		*((volatile int*)&dFooReadyPtr[id]) = 1;
}

/*
 Consumer function.

 Preceding a threadfence (memory barrier) with a volatile yields an acquire pattern.
 Preceding a threadfence (memory barrier) with an atomic yields an acquire pattern.
 Note however, that neither of these options is ideal. For one, we combine two 
 operations to achieve a certain behavior. Second, threadfence is a general memory
 barrier, and is thus stronger than it may have to be (e.g., acquire barrier only).
 Volta+ actually have support for memory coherency models with proper acquire /
 release semantics, which are exposed to the programmer via cuda::std::atomic in
 libcu++.
*/
template <bool ATOMIC>
__device__ void ConsumeFoo(unsigned int id, const float* dFooPtr, int* dFooReadyPtr)
{
	if (ATOMIC)
		while (atomicAdd(&dFooReadyPtr[id], 0) == 0);
	else
		while (*((volatile int*)&dFooReadyPtr[id]) == 0);

	__threadfence();

	printf("Consumer %d thinks Pi is: %f\n", id, dFooPtr[id]);
}

// Launch either version of a safe producer / consumer scenarios
template <unsigned int N, bool ATOMIC>
__global__ void ProducerConsumer(float* dFooPtr, int* dFooReadyPtr)
{
	int id = (blockIdx.x * blockDim.x + threadIdx.x);

	if (id < N)
		ProduceFoo<ATOMIC>(id, dFooPtr, dFooReadyPtr);
	else
		ConsumeFoo<ATOMIC>(id - N, dFooPtr, dFooReadyPtr);
}

/*
 As we have seen before, although we didnt explicilty mention it,
 using a syncthreads inside a block is sufficient to make sure that
 the other threads can observe the data that was previously written
 by another thread in the block. Here we illustrate this again, 
 with a simple, safe producer / consumer setup, where syncthreads
 ensures ordering of operations and visibility of the data for all
 threads in the block.
*/
__global__ void ProducerConsumerShared()
{
	extern __shared__ float sFoo[];

	if (threadIdx.x < blockDim.x/2)
	{
		float pi = samplesutil::GregoryLeibniz(10'000'000);
		sFoo[threadIdx.x] = pi;
	}
	// Synchronize threads in block AND ensure memory access ordering among them
	__syncthreads();
	if (threadIdx.x >= blockDim.x / 2)
	{
		int cId = threadIdx.x - blockDim.x / 2;
		printf("Comsumer %d thinks Pi is %f\n", cId, sFoo[cId]);
	}
}

int main()
{
	std::cout << "==== Sample 13 - Memory Fences ====\n" << std::endl;
	/*
	 So far, we have ignored the problem of memory fencing, which
	 is relevant in multi-threaded applications. We can exchange 
	 information securely via atomic variables, however when we store
	 data in bulk or need to ensure a particular ordering of observed
	 events, for instance in a producer/consumer scenario, we need clear 
	 orderings of data accesses that are definite for all involved threads. 
	 For threads within a block, this is trivially achieved by using 
	 syncthreads. For establishing orderings across blocks, CUDA offers
	 the __threadfence operation. This can be necessary, because the default 
	 atomicXXX operations of CUDA only give us "RELAXED" semantics, i.e., 
	 they have no synchronization effect on other memory. However, combining
	 a thread fence with relaxed atomics can---much like in C++11---give us 
	 acquire / release semantics. 
	 
	 At its core, threadfence is a general memory barrier, which makes sure 
	 that all writes below it occur after all writes above it, and that all 
	 reads below it occur after all reads above it. However, there are some
	 intricacies that make the safe use of threadfence a little tricky. 
	 Understanding all possible scenarios is complex task, and may not be worth 
	 the effort, since modern CUDA offers better alternatives (see material and 
	 samples for CUDA standard library). A basic recipe for safely using
	 __threadfence is as part of a release-acquire pattern. The PTX ISA states
	 that a __threadfence, followed by an atomic or volatile memory operation,
	 yields a release pattern, while a __threadfence preceding an atomic or
	 volatile memory operation yields an acquire pattern. With these patterns,
	 we can for instance solve the producer / consumer scenario by using flags
	 that indicate when data is ready, and securing access to them with proper
	 acquire / release behavior. 

	 Expected output: 

		Producer / consumer pair in same block
		Comsumer 15 thinks Pi is 3.141597
		Comsumer 16 thinks Pi is 3.141597
		Comsumer 0 thinks Pi is 3.141597
		...
		(or similar)

		Producer / consumer pair with volatile + threadfence
		Consumer 4 thinks Pi is: 3.141597
		...
		(or similar)

		Producer / consumer pair with volatile + atomic
		Consumer 4 thinks Pi is: 3.141597
		...
		(or similar)
	*/

	constexpr unsigned int N = 8;
	constexpr unsigned int blockSize = 4;
	
	// Compute how many producer / consumer blocks should be launched
	unsigned int numBlocks = N / blockSize;

	// Run producer / consumer scenario inside a single block (simple)
	std::cout << "\nProducer / consumer pair in same block" << std::endl;
	ProducerConsumerShared<<<1, 34, 34 * sizeof(float)>>>();
	cudaDeviceSynchronize();

	// Allocate and initialize mmeory for global producer / consumer scenario
	float* dFooPtr;
	int* dFooReadyPtr;
	cudaMalloc((void**)&dFooPtr, sizeof(float) * N);
	cudaMalloc((void**)&dFooReadyPtr, sizeof(int) * N);
	cudaMemset(dFooPtr, 0, sizeof(float) * N);
	cudaMemset(dFooReadyPtr, 0, sizeof(int) * N);

	// Producer / consumer scenario across blocks in global memory, using volatile + threadfence
	std::cout << "\nProducer / consumer pair with volatile + threadfence" << std::endl;
	ProducerConsumer<N, false><<<numBlocks * 2, blockSize>>>(dFooPtr, dFooReadyPtr);
	cudaDeviceSynchronize();

	// Producer / consumer scenario across blocks in global memory, using atomic + threadfence
	std::cout << "\nProducer / consumer pair with atomic + threadfence" << std::endl;
	ProducerConsumer<N, true><<<numBlocks * 2, blockSize>>>(dFooPtr, dFooReadyPtr);
	cudaDeviceSynchronize();

	return 0;
}

/*
Exercises:
1) TRY to write a program where one thread reliably observes writes in the WRONG
order, due to lack of threadfence (e.g., in your code one thread sets A from 0 
to 1, followed by setting B from 0 to 1, but another thread observes A = 0, B = 1,
or something similar). To do this, you may want to make sure those threads run in 
different blocks, preferably even on different SMs, and communicate via global 
memory, try atomics and volatiles. If you can't manage, document your best attempt. 
*/