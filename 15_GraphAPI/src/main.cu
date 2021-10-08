#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>
#include "../../shared/include/utility.h"

// A simple kernel function to keep threads busy for a while
__global__ void busy()
{
	samplesutil::WasteTime(1'000'000'000ULL);
	printf("I'm awake!\n");
}

void runTasksSequentially(unsigned int numTasks)
{
	// We use cudaStreamPerThread here. It makes no difference
	// for the program flow because we are only single-threaded
	// anyway, but capturing the application-wide default stream 
	// cudaStreamLegacy is not permitted by the graph API.

	for (int i = 0; i < numTasks; i++)
		busy << <1, 1, 0, cudaStreamPerThread >> > ();
}

void runTasksWithStreams(unsigned int numTasks)
{
	/*
	This stream-based function can be directly captured with graph API.
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
		busy << <1, 1, 0, streams[i] >> > ();
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
	// Create a graph for recording GPU commands
	cudaGraph_t graph;
	cudaGraphCreate(&graph, 0);

	// Record a graph, assuming that functions start from thread's default stream
	cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
	func(params...);
	cudaStreamEndCapture(cudaStreamPerThread, &graph);

	// Turn the recorded graph into an executable instance
	cudaGraphExec_t instance;
	cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	// The recorded graph is no longer needed
	cudaGraphDestroy(graph);
	return instance;
}

cudaGraphExec_t buildGraphForParallelTasks(unsigned int numTasks)
{
	// Set up a graph from scratch
	cudaGraph_t graph;
	cudaGraphCreate(&graph, 0);

	// Create a node for each kernel in the graph, with grid config and parameters
	std::vector<cudaGraphNode_t> nodes(numTasks);
	cudaKernelNodeParams params = { reinterpret_cast<void*>(busy), {1,1,1}, {1,1,1}, 0, nullptr, nullptr };

	// Add them to the graph. This simple setup has no dependencies, passing nullptr
	for (int i = 0; i < numTasks; i++)
		cudaGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &params);

	// Create executable graph, destroy manually built graph
	cudaGraphExec_t instance;
	cudaGraphInstantiate(&instance, graph, 0, 0, 0);
	cudaGraphDestroy(graph);
	return instance;
}

int main()
{
	std::cout << "==== Sample 15 - Graph API ====\n" << std::endl;
	/*
	The graph API enables the creation of well-defined structures that
	encode the types, parameters and dependencies of instructions that
	the GPU should process. By preparing this information, developers can
	decouple the definition and execution of the parallel workload. The
	driver is then free to optimize its execution. Graphs may be created
	either by setting up graphs manually from scratch or by recording
	already available code, which may occasionally require modifications:
	CUDA graphs usually do not include synchronization methods with the
	CPU. This means that waiting actions (e.g., until all streams have
	finished) must be modelled via dependencies/events instead.

	Expected output: 5 x TASKS "I'm awake\n", first two groups launching
	sequentially, the last three groups running concurrently. 
	*/

	constexpr int TASKS = 4;

	std::cout << "Launching multiple tasks sequentially" << std::endl;
	// Launching multiple tasks as kernels one after the other
	runTasksSequentially(TASKS);
	cudaDeviceSynchronize();

	std::cout << "Running recorded graph from existing sequential code" << std::endl;
	// Recording a graph from the existing sequential code and launching its instance
	cudaGraphExec_t recordedSequential = recordGraphFromFunction(runTasksSequentially, TASKS);
	cudaGraphLaunch(recordedSequential, 0);
	cudaDeviceSynchronize();

	std::cout << "Launching multiple tasks with streams" << std::endl;
	// Launching multiple tasks in multiple streams
	runTasksWithStreams(TASKS);
	cudaDeviceSynchronize();

	std::cout << "Running recorded graph from existing stream-based code" << std::endl;
	// Recording a graph from the existing stream-based code, launching instance
	cudaGraphExec_t recordedStreams = recordGraphFromFunction(runTasksWithStreams, TASKS);
	cudaGraphLaunch(recordedStreams, 0);
	cudaDeviceSynchronize();

	std::cout << "Running manually-built graph that behaves like streams" << std::endl;
	// Example for building a scratch manually without recording
	cudaGraphExec_t instanceBuilt = buildGraphForParallelTasks(TASKS);
	cudaGraphLaunch(instanceBuilt, 0);
	cudaDeviceSynchronize();

	return 0;
}

/*
Exercises:
1) Manually build and instantiate a working CUDA graph that includes a host-side 
function node and demonstrate what it does.
2) Manually build and instantiate a working graph that includes a different kind 
of node that does not yet occur in this application or in exercise 1).
3) Given a particular CUDA graph, create a simple GraphViz graph of its structure.
Use it to sketch out the graphs created by this source code. 
*/