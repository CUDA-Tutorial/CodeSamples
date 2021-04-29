#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Simulate a complex task, but actually only compute a square
__global__ void PerformComplexTask(float input, float* __restrict result)
{
	const int start = clock();
	while ((clock() - start) < 1'000'000'000);
	*result = input * input;
}

int main()
{
	std::cout << "==== Sample 09 - Pinned Memory ====\n" << std::endl;
	/*
	Pinned memory becomes relevant once we start using streams
	and memory transfer enters the mix. The default memcpy operation
	cudaMemcpy is, by default synchronous, i.e., when it is called,
	the CPU will stall until the memcpy has finished. However, in 
	many cases we don't want this. Consider the example below, 
	where we use several streams to compute some expensive result
	from different inputs. For each 'task', we launch a kernel to
	a separate stream, followed by a memcpy of the result back to
	the GPU. 

	Ideally, we would like the memory transfers to overlap with kernels
	that run in different streams. But if we use cudaMemcpy, the kernel 
	calls will execute sequentially, because each cudaMemcpy implicitly 
	synchronizes with the CPU. However, there is a different memory 
	transfer function of the name cudaMemcpyAsync, which also takes an 
	additional stream parameter in which to run. However, using this 
	function alone is not enough to overlap memory transfer and kernels.
	To perform asynchronous memcpy between the device and the host, CUDA 
	must be sure that the host memory is available in main memory. We 
	can guarantee this by allocating memory with cudaMallocHost. This is
	so-called "pinned" memory, which may never be moved or swapped out.
	If we use pinned memory and cudaMemcpyAsync, then copies and kernels
	that run in different streams are free to overlap.

	Expected output: slow performance for all combinations that are not
	pinned memory and asynchronous copy, due to implicit synchronization
	prevening concurrent execution of kernels.
	*/

	constexpr unsigned int TASKS = 4;

	// Allocate result values for GPU to write to
	float* dResultsPtr;
	cudaMalloc((void**)&dResultsPtr, sizeof(float) * TASKS);

	// Generate necessary streams
	cudaStream_t streams[TASKS];
	for (cudaStream_t& s : streams)
		cudaStreamCreate(&s);

	// Two CPU-side memory ranges: one regular and one pinned
	float results[TASKS], * results_pinned;
	cudaMallocHost((void**)&results_pinned, sizeof(float) * TASKS);

	// We run the tasks with regular/async memcpy
	enum class CPYTYPE { MEMCPY, MEMCPYASYNC };
	// We run the tasks with regular/pinned memory
	enum class MEMTYPE { REGULAR, PINNED};

	for (auto mem : { MEMTYPE::REGULAR, MEMTYPE::PINNED })
	{
		float* dst = (mem == MEMTYPE::PINNED ? results_pinned : results);

		for (auto cpy : { CPYTYPE::MEMCPY, CPYTYPE::MEMCPYASYNC })
		{
			std::cout << "Performing tasks with " << (mem == MEMTYPE::PINNED ? "pinned memory" : "regular memory");
			std::cout << " and " << (cpy == CPYTYPE::MEMCPYASYNC ? "asynchronous" : "regular") << " copy" << std::endl;

			// Synchronize to get adequate CPU time measurements
			cudaDeviceSynchronize();
			auto before = std::chrono::system_clock::now();

			for (int i = 0; i < TASKS; i++)
			{
				// Unnecessarily slow kernel
				PerformComplexTask<<<1, 1, 0, streams[i]>>>(i+1, dResultsPtr+i);
				// Use either regular or asynchronous copy for reading back results
				if (cpy == CPYTYPE::MEMCPYASYNC)
					cudaMemcpyAsync(&dst[i], dResultsPtr+i, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
				else
					cudaMemcpy(&dst[i], dResultsPtr+i, sizeof(float), cudaMemcpyDeviceToHost);
			}

			// Synchronize to get adequate CPU time measurements
			cudaDeviceSynchronize();
			auto after = std::chrono::system_clock::now();

			// Print computed results and total CPU-side runtime
			for (int i = 0; i < TASKS; i++)
				std::cout << i+1 << " squared = " << results[i] << std::endl;
			std::cout << "Time: " << std::chrono::duration_cast<std::chrono::duration<float>>(after-before).count() << "s\n\n";
		}
	}

	// Clean up streams
	for (cudaStream_t& s : streams)
		cudaStreamDestroy(s);

	// Pinned memory should be freed with cudaFreeHost
	cudaFreeHost(results_pinned);
}
