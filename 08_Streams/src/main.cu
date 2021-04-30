#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// A simple kernel function to keep threads busy for a while
__global__ void busy()
{
	const int start = clock();
	while ((clock() - start) < 1'000'000'000);
	printf("I'm awake!\n");
}

int main()
{
	std::cout << "==== Sample 08 - Streams ====\n" << std::endl;
	/*
	 Expected output: "I'm awake!\n" x 4 x KERNEL_CALLS + 4

	 If you watch the output carefully or analyze the execution of 
	 this program with NVIDIA Nsight Systems, it should show that the 
	 first group of kernels run consecutively, while the second and 
	 third group run in parallel. 
	 
	 Finally, there should be two kernels running sequentially,
	 followed by two kernels running in parallel.
	*/

	constexpr unsigned int KERNEL_CALLS = 2;

	std::cout << "Running sequential launches" << std::endl;
	// Launch the same kernel several times in a row
	for (unsigned int i = 0; i < KERNEL_CALLS; i++)
		busy<<<1, 1>>>();
	// Synchronize before continuing to get clear separation in Nsight
	cudaDeviceSynchronize();

	std::cout << "\nRunning launches in streams" << std::endl;
	// Allocate one stream for each kernel to be launched
	cudaStream_t streams[KERNEL_CALLS];
	for (cudaStream_t& s : streams)
	{
		// Create stream and launch kernel into it
		cudaStreamCreate(&s);
		busy<<<1, 1, 0, s>>>();
	}
	// Destroy all streams (implicitly waits until each has finished)
	for (cudaStream_t& s : streams)
		cudaStreamDestroy(s);
	cudaDeviceSynchronize();

	/*
	If we don't specify a stream, then the kernel is launched into the default 
	stream. Also, many operations like cudaDeviceSynchronize and 
	cudaStreamSynchronize are submitted to the default stream. Usually, only a 
	single default stream is defined per application, meaning that if you don't 
	specify streams, you will not be able to benefit from kernels running 
	concurrently. Hence, any elaborate CUDA application should be using streams. 
	
	However, if the task can be cleanly separated into CPU threads, there is another 
	option: using per-thread default streams. Each thread will use its own default
	stream if we pass the built-in value cudaStreamPerThread as the stream to use.
	Kernels can then run concurrently on the GPU by creating multiple CPU threads.
	Alternatively, you may set the compiler option "--default-stream per-thread". 
	This way, CPU threads will use separate default streams if none are specified.
	*/
	std::cout << "\nRunning threads with different default streams" << std::endl;

	// Create mutex, condition variable and counter for communication
	std::mutex mutex;
	std::condition_variable cv;
	unsigned int kernelsLaunched = 0;
	// Allocate sufficient number of threads
	std::thread threads[KERNEL_CALLS];
	// Create a separate thread for each kernel call (task)
	for (std::thread& t : threads)
	{
		t = std::thread([&mutex, &cv, &kernelsLaunched] {
			// Launch kernel to thread's default stream
			busy<<<1, 1, 0, cudaStreamPerThread>>>();
			/*
			 Make sure all kernels are submitted before synchronizing,
			 because cudaStreamSynchronize goes into the default 0 stream:
			 busy<1> -> sync<0>(1) -> busy<2> -> sync<0>(2)... serializes.
			 busy<1> -> busy<2> -> sync<0>(1) -> sync<0>(2)... parallelizes.
			*/
			std::unique_lock<std::mutex> lock(mutex);
			++kernelsLaunched;
			cv.wait(lock, [&kernelsLaunched] { return kernelsLaunched == KERNEL_CALLS; });
			cv.notify_all();
			// Synchronize to wait for printf output
			cudaStreamSynchronize(cudaStreamPerThread);
		});
	}
	// Wait for all threads to finish launching their kernels in individual streams
	std::for_each(threads, threads + KERNEL_CALLS, [](std::thread& t) {t.join(); });

	/*
	By default, custom created streams will implicitly synchronize with the 
	default stream. Consider, e.g., a kernel A running in a custom stream, 
	followed by a kernel B in the default stream. If we use cudaStreamCreate
	as above, then A will end before B starts. Alternatively, we may create 
	custom streams with the flag cudaStreamNonBlocking. In this case, the 
	custom stream will not synchronize with the default stream anymore. 
	*/
	cudaStream_t customRegular, customNonblocking;
	cudaStreamCreate(&customRegular);
	cudaStreamCreateWithFlags(&customNonblocking, cudaStreamNonBlocking);

	auto testAB = [](const char* kind, cudaStream_t stream) {
		std::cout << "\nLaunching A (custom) -> B (default) with " << kind << " custom stream" << std::endl;
		busy<<<1, 1, 0, stream>>>();
		busy<<<1, 1>>>();
		cudaDeviceSynchronize();
	};

	testAB("regular", customRegular);
	testAB("non-blocking", customNonblocking);

	// Clean up generated streams
	cudaStreamDestroy(customRegular);
	cudaStreamDestroy(customNonblocking);

	return 0;
}
