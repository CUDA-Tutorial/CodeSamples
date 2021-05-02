#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cuda/std/atomic>

__host__ __device__ void reduceBasic(int tId, int numThreads, int N, const int* input, cuda::std::atomic<int>* result)
{
	if (tId >= N)
		return;

	int perThread = N / numThreads;
	int myStart = perThread * tId;
	int myEnd = (tId == numThreads - 1) ? N : myStart + perThread;

	for (int i = myStart; i < myEnd; i++)
	{
		int val = input[i];
		result->fetch_add(val);
	}
}

__global__ void launchReduceBasic(int N, const int* input, cuda::std::atomic<int>* result)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	reduceBasic(tId, N, N, input, result);
}

int main()
{
	/*
	We use integers in this example. Float atomics are only part
	of the standard in C++20. Soon!
	*/
	constexpr int N = 1<<16;
	constexpr int cpuNumThreads = 4;
	constexpr int gpuBlockSize = 256;

	int* mNumbers;
	cudaMallocManaged((void**)&mNumbers, sizeof(int) * N);

	std::default_random_engine eng(42);
	std::uniform_int_distribution<int> dist(10, 42);

	cuda::std::atomic<int>* mResults;
	cudaMallocManaged((void**)&mResults, 2 * sizeof(cuda::std::atomic<int>));
	new (mResults) cuda::std::atomic<int>(0);
	new (mResults+1) cuda::std::atomic<int>(0);

	for (int i = 0; i < N; i++)
		mNumbers[i] = dist(eng);

	std::vector<std::thread> threads(cpuNumThreads);
	for (int i = 0; i < threads.size(); i++)
		threads[i] = std::thread(reduceBasic, i, cpuNumThreads, N, mNumbers, &mResults[0]);

	for (std::thread& t : threads)
		t.join();

	std::cout << "Reduction result CPU: " << mResults[0] << "\n" << std::endl;

	launchReduceBasic<<<(N + gpuBlockSize - 1) / gpuBlockSize, gpuBlockSize>>>(N, mNumbers, &mResults[1]);
	cudaDeviceSynchronize();

	std::cout << "Reduction result GPU: " << mResults[1] << "\n" << std::endl;

	cudaFree(mNumbers);
	cudaFree(mResults);
}