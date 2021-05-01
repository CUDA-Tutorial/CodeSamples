#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

// Simulate a complex task, but actually only compute a square
__global__ void PerformComplexTask(float input, float* __restrict result)
{
	const int start = clock();
	while ((clock() - start) < 100'000'000);
	*result = input * input;
}

int main()
{
	std::cout << "==== Sample 11 - Pinned Memory ====\n" << std::endl;
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
	synchronizes the default stream with the CPU, and all basic streams 
	are synchronized with the default stream. However, there is a different 
	memory transfer function of the name cudaMemcpyAsync, which also takes 
	an additional stream parameter in which to run. However, using this 
	function alone is not enough to overlap memory transfer and kernels.
	To perform asynchronous memcpy between the device and the host, CUDA 
	must be sure that the host memory is available in main memory. We 
	can guarantee this by allocating memory with cudaMallocHost. This is
	so-called "pinned" memory, which may never be moved or swapped out.
	If we use pinned memory and cudaMemcpyAsync, then copies and kernels
	that run in different streams are free to overlap.

	Expected output: slow performance for all combinations that are not
	pinned memory and asynchronous copy, due to implicit synchronization
	preventing concurrent execution of kernels.

	You may also try to make the streams non-blocking! In this case, you
	can expect wrong results for cudaMemcpy: the default stream won't
	wait for the custom streams running the kernels to finish before it
	starts copying.
	*/

	constexpr unsigned int TASKS = 4;

	// Allocate result values for GPU to write to
	float* dResultsPtr;
	cudaMalloc((void**)&dResultsPtr, sizeof(float) * TASKS);

	// Generate necessary streams and events
	cudaStream_t streams[TASKS];
	cudaEvent_t events[TASKS];
	for (int i = 0; i < TASKS; i++)
	{
		cudaStreamCreate(&streams[i]);
		cudaEventCreate(&events[i]);
	}

	// Two CPU-side memory ranges: one regular and one pinned
	float results[TASKS], * results_pinned;
	cudaMallocHost((void**)&results_pinned, sizeof(float) * TASKS);

	// We run the tasks with regular/async memcpy
	enum class CPYTYPE { MEMCPY, MEMCPYASYNC };
	// We run the tasks with regular/pinned memory
	enum class MEMTYPE { REGULAR, PINNED};

	for (auto cpy : { CPYTYPE::MEMCPY, CPYTYPE::MEMCPYASYNC })
	{
		for (auto mem : { MEMTYPE::REGULAR, MEMTYPE::PINNED })
		{
			float* dst = (mem == MEMTYPE::PINNED ? results_pinned : results);

			std::cout << "Performing tasks with " << (mem == MEMTYPE::PINNED ? "pinned memory" : "regular memory");
			std::cout << " and " << (cpy == CPYTYPE::MEMCPYASYNC ? "asynchronous" : "regular") << " copy" << std::endl;

			// Reset GPU result
			cudaMemset(dResultsPtr, 0, sizeof(float) * TASKS);

			// Synchronize to get adequate CPU time measurements
			cudaDeviceSynchronize();
			const auto before = std::chrono::system_clock::now();

			for (int i = 0; i < TASKS; i++)
			{
				// Unnecessarily slow kernel
				PerformComplexTask<<<1, 1, 0, streams[i]>>>(i+1, dResultsPtr+i);
				// Use either regular or asynchronous copy for reading back results
				if (cpy == CPYTYPE::MEMCPYASYNC)
					cudaMemcpyAsync(&dst[i], dResultsPtr+i, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
				else
					cudaMemcpy(&dst[i], dResultsPtr + i, sizeof(float), cudaMemcpyDeviceToHost);
			}

			// Wait for results being copied back
			for (int i = 0; i < TASKS; i++)
			{
				// Wait for the current stream
				cudaStreamSynchronize(streams[i]);

				// Evaluate result and print
				if (results[i] != (i + 1) * (i + 1))
					std::cout << "Task failed or CPU received wrong value!" << std::endl;
				else
					std::cout << "Finished task " << i << ", produced output: " << results[i] << std::endl;
			}

			const auto after = std::chrono::system_clock::now();
			std::cout << "Time: " << std::chrono::duration_cast<std::chrono::duration<float>>(after-before).count() << "s\n\n";
		}
	}

	// Clean up streams
	for (cudaStream_t& s : streams)
		cudaStreamDestroy(s);

	// Pinned memory should be freed with cudaFreeHost
	cudaFreeHost(results_pinned);
}
