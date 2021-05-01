#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

// A simple kernel function to keep threads busy for a while
__global__ void busy()
{
	const int start = clock();
	while ((clock() - start) < 1'000'000'000);
	printf("I'm awake!\n");
}

void runTasksSequentially(unsigned int numTasks)
{
	// We use cudaStreamPerThread here. It makes no difference
	// for the program flow because we are only single-threaded
	// anyway, but capturing the application-wide default stream 
	// cudaStreamLegacy is not permitted by the graph API.

	for (int i = 0; i < numTasks; i++)
		busy<<<1, 1, 0, cudaStreamPerThread>>>();
}

void runTasksWithStreams(unsigned int numTasks)
{
	/* 
	This stream-based function that can be directly captured with graph API.
	Events are used to encode dependencies / start / end of capture.
	*/
	std::vector<cudaStream_t> streams(numTasks);
	std::vector<cudaEvent_t> finished(numTasks);
	for (int i = 0; i < numTasks; i++)
	{
		cudaStreamCreate(&streams[i]);
		cudaEventCreate(&finished[i]);
	}	
	// We need an additional event to represent the capture start
	cudaEvent_t start;
	cudaEventCreate(&start);
	// Immediately record the starting event so other streams can connect to it
	cudaEventRecord(start, cudaStreamPerThread);
	// All other streams must connect to origin stream via event to get captured
	for (int i = 0; i < numTasks; i++)
	{
		// Establish dependency / connection to origin (is now included in capture)
		cudaStreamWaitEvent(streams[i], start);
		// Run actual task (kernel) in stream
		busy<<<1, 1, 0, streams[i]>>>();
		// Record end event of this stream so origin can wait on it
		cudaEventRecord(finished[i], streams[i]);
	}
	// Origin stream waits until all custom streams have finished their task
	for (int i = 0; i < numTasks; i++)
	{
		cudaStreamWaitEvent(cudaStreamPerThread, finished[i]);
		cudaStreamDestroy(streams[i]);
	}
}

template <typename T, typename ...P>
cudaGraphExec_t recordGraphFromFunction(const T& func, P ...params)
{
	cudaGraph_t graph;
	cudaGraphCreate(&graph, 0);

	cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
	func(params...);
	cudaStreamEndCapture(cudaStreamPerThread, &graph);

	cudaGraphExec_t instance;
	cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	cudaGraphDestroy(graph);
	return instance;
}

cudaGraphExec_t buildGraphForParallelTasks(unsigned int numTasks)
{
	cudaGraph_t graph;
	cudaGraphCreate(&graph, 0);

	std::vector<cudaGraphNode_t> nodes(numTasks);
	cudaKernelNodeParams params = { busy, {1,1,1}, {1,1,1}, 0, nullptr, nullptr };

	for (int i = 0; i < numTasks; i++)
		cudaGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &params);

	cudaGraphExec_t instance;
	cudaGraphInstantiate(&instance, graph, 0, 0, 0);
	cudaGraphDestroy(graph);

	return instance;
}

int main()
{
	std::cout << "==== Sample 15 - Graph API ====\n" << std::endl;
	/*

	The CUDA graph does NOT include synchronization methods with the CPU!
	This means that waiting actions (e.g., until all streams have finished)
q	must be modelled via dependencies/events instead.
	*/

	constexpr int TASKS = 4;

	std::cout << "Launching multiple tasks sequentially" << std::endl;

	runTasksSequentially(TASKS);
	cudaDeviceSynchronize();

	std::cout << "Running recorded graph from existing sequential code" << std::endl;

	cudaGraphExec_t recordedSequential = recordGraphFromFunction(runTasksSequentially, TASKS);
	cudaGraphLaunch(recordedSequential, 0);
	cudaDeviceSynchronize();

	std::cout << "Launching multiple tasks with streams" << std::endl;

	runTasksWithStreams(TASKS);
	cudaDeviceSynchronize();

	std::cout << "Running recorded graph from existing stream-based code" << std::endl;

	cudaGraphExec_t recordedStreams = recordGraphFromFunction(runTasksWithStreams, TASKS);
	cudaGraphLaunch(recordedStreams, 0);
	cudaDeviceSynchronize();

	std::cout << "Running manually-built graph that behaves like streams" << std::endl;

	cudaGraphExec_t instanceBuilt = buildGraphForParallelTasks(TASKS);
	cudaGraphLaunch(instanceBuilt, 0);
	cudaDeviceSynchronize();

	return 0;
}