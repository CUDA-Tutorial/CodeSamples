#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cuda/std/atomic>

/*
 Basic, architecture-agnostic reduction, using global atomics.
 Uses portable cuda::std::atomics. Note that even though this code is portable,
 it may not necessarily give the best performance. cuda::std::atomics have
 system-wide (CPU + GPU) scope. If an algorithm is sure to run on the GPU, better
 performance may be achieved using cuda::atomics, which take an additional parameter
 "thread_scope" (e.g., "device" for global, "block" for shared memory atomics).
*/
__host__ __device__ void reduceAtomic(int tId, int numThreads, int N, const int* input, cuda::std::atomic<int>* result)
{
	if (tId >= N)
		return;

	// Compute input portion to be handled by each thread
	int perThread = N / numThreads;
	int myStart = perThread * tId;
	int myEnd = (tId == numThreads - 1) ? N : myStart + perThread;

	// For each value in the assigned portion, atomically add it to accumulated sum.
	for (int i = myStart; i < myEnd; i++)
		result->fetch_add(input[i], cuda::std::memory_order_relaxed);
}

__global__ void completeReductionGPU(int N, const int* input, cuda::std::atomic<int>* result)
{
	// Launchpad code for calling the architecutre-agnostic reduction function
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	reduceAtomic(tId, N, N, input, result);
}

template<unsigned int NUM_THREADS>
__host__ void completeReductionCPU(int N, int* mNumbers, cuda::std::atomic<int>* result)
{
	/*
	 Simple multi-threaded launch for computing the reduction 
	 in parallel. This function also makes sure to join on the
	 threads. Otherwise, we would have to be careful not to
	 overlap CPU access to managed memory and GPU execution.
	 Each thread uses the architecture-agnostic implementation
	 to compute part of the full reduction.
	*/
	std::vector<std::thread> threads(NUM_THREADS);
	for (int i = 0; i < threads.size(); i++)
		threads[i] = std::thread(reduceAtomic, i, NUM_THREADS, N, mNumbers, result);
	for (std::thread& t : threads)
		t.join();
}

int main()
{
	std::cout << "==== Sample 18 - Standard Library ====\n" << std::endl;
	/*
	The libcu++ standard library allows us to make code more portable. 
	Users can use familiar concepts from programming for the CPU and
	apply them with minimal changes on the GPU as well. In this 
	example, we show a method for parallel reduction where the same
	reduction function can be executed by a thread on the CPU or the
	GPU thanks to the support for std:: atomics.

	We use integers in this example. Float atomics are only part
	of the standard in C++20. Should be widely available soon! Once 
	more, the libcu++ standard library is much more powerful than 
	we can show with just a few samples. We encourage you to check
	out the documentation for libcu++, as well as related talks for
	more examples of use cases. 

	Expected output: the result of a reduction with random integers,
	once computed on the CPU and once on the GPU, both of them 
	yielding the same accumulated value. 
	*/

	// Define the number of inputs to reduce, and number of CPU threads
	constexpr int N = 1 << 16, CPUThreads = 4;

	// Allocate managed memory and fill it with random numbers
	int* mNumbers;
	std::default_random_engine eng(42);
	std::uniform_int_distribution<int> dist(10, 42);
	cudaMallocManaged((void**)&mNumbers, sizeof(int) * N);
	std::for_each(mNumbers, mNumbers + N, [&dist, &eng](int& v) { v = dist(eng); });
	// Allocate managed memory for the computed recution results (CPU/GPU)
	cuda::std::atomic<int>* mResults;
	cudaMallocManaged((void**)&mResults, 2 * sizeof(cuda::std::atomic<int>));
	mResults[0] = mResults[1] = 0;

	// Launch the reduction with the given number of CPU threads
	completeReductionCPU<CPUThreads>(N, mNumbers, &mResults[0]);
	// Launch the reduction on the GPU with as many threads as there are inputs
	completeReductionGPU<<<(N + 255) / 256, 256>>>(N, mNumbers, &mResults[1]);
	cudaDeviceSynchronize();

	// Output both results
	std::cout << "Reduction result CPU: " << mResults[0] << "\n" << std::endl;
	std::cout << "Reduction result GPU: " << mResults[1] << "\n" << std::endl;

	// finally, release the managed memory for the inputs and results
	cudaFree(mNumbers);
	cudaFree(mResults);
}

/*
Exercises:
1) The CUDA standard library is continuously being expanded. Check out
their documentation and use an include for one of the recent features
and demonstrate it.
2) Write a simple kernel with a single block that frequently updates a
single cuda::atomic variable. For performance reasons, it should be one with
thread_scope "block".
3) Try to show that there is a performance difference in 2) between using
the default cuda::std::atomic and the cuda::atomic with thread_scope block.
*/