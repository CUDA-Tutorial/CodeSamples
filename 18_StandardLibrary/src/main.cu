#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cuda/std/atomic>

__host__ __device__ void reduceAtomic(int tId, int numThreads, int N, const int* input, cuda::std::atomic<int>* result)
{
	if (tId >= N)
		return;

	int perThread = N / numThreads;
	int myStart = perThread * tId;
	int myEnd = (tId == numThreads - 1) ? N : myStart + perThread;

	for (int i = myStart; i < myEnd; i++)
		result->fetch_add(input[i]);
}

__global__ void launchReductionGPU(int N, const int* input, cuda::std::atomic<int>* result)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	reduceAtomic(tId, N, N, input, result);
}

template<unsigned int NUM_THREADS>
__host__ void launchReductionCPU(int N, int* mNumbers, cuda::std::atomic<int>* result)
{
	std::vector<std::thread> threads(NUM_THREADS);
	for (int i = 0; i < threads.size(); i++)
		threads[i] = std::thread(reduceAtomic, i, NUM_THREADS, N, mNumbers, result);
	for (std::thread& t : threads)
		t.join();
}

int main()
{
	/*
	We use integers in this example. Float atomics are only part
	of the standard in C++20. Should be widely available soon!
	*/
	constexpr int N = 1 << 16, CPUThreads = 4;

	std::default_random_engine eng(42);
	std::uniform_int_distribution<int> dist(10, 42);

	int* mNumbers;
	cudaMallocManaged((void**)&mNumbers, sizeof(int) * N);
	std::for_each(mNumbers, mNumbers + N, [&dist, &eng](int& v) { v = dist(eng); });

	cuda::std::atomic<int>* mResults;
	cudaMallocManaged((void**)&mResults, 2 * sizeof(cuda::std::atomic<int>));
	mResults[0] = mResults[1] = 0;

	launchReductionCPU<CPUThreads>(N, mNumbers, &mResults[0]);
	launchReductionGPU<<<(N + 255) / 256, 256>>>(N, mNumbers, &mResults[1]);
	cudaDeviceSynchronize();

	std::cout << "Reduction result CPU: " << mResults[0] << "\n" << std::endl;
	std::cout << "Reduction result GPU: " << mResults[1] << "\n" << std::endl;

	cudaFree(mNumbers);
	cudaFree(mResults);
}