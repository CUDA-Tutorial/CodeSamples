#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "../../shared/include/utility.h"

// A kernel that wastes some time
__global__ void SlowKernel()
{
    samplesutil::WasteTime(1'000'000'000ULL);
}

__device__ int dFoo;

// A kernel that only sets dFoo
__global__ void SetFoo(int foo)
{
    dFoo = foo;
}

// A kernel that prints dFoo
__global__ void PrintFoo()
{
    printf("foo: %d\n", dFoo);
}

int main()
{
    std::cout << "==== Sample 10 - Events ====\n" << std::endl;
    /*
     Using events to measure time and communicate across streams.

     Expected output: 
     1) Unrealistically short time with chrono measurements without syncing, 
     similar times for chrono with syncing and when using CUDA events.
     2) foo: 42
    */
    using namespace std::chrono_literals;
    using namespace std::chrono;

    // Create CUDA events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Synchronize GPU with CPU to capture adequate time
    cudaDeviceSynchronize();
    auto before = std::chrono::system_clock::now();

    // Record start directly before first relevant GPU command
    cudaEventRecord(start);
    // Launch a light-weight GPU kernel and heavy GPU kernel
    SetFoo<<<1,1>>>(0);
    SlowKernel<<<1,1>>>();
    // Record end directly after last relevant GPU command
    cudaEventRecord(end);
    // Also measure CPU time after last GPU command, without synching
    auto afterNoSync = std::chrono::system_clock::now();

    // Synchronize CPU and GPU
    cudaDeviceSynchronize();
    // Measure CPU time after last GPU command, with synching
    auto afterSync = std::chrono::system_clock::now();

    // Print measured CPU time without synchronization
    float msCPUNoSync = 1000.f * duration_cast<duration<float>>(afterNoSync - before).count();
    std::cout << "Measured time (chrono, no sync): " << msCPUNoSync << "ms\n";

    // Print measured CPU time with synchronization
    float msCPUSync = 1000.f * duration_cast<duration<float>>(afterSync - before).count();
    std::cout << "Measured time (chrono, sync): " << msCPUSync << "ms\n";

    // Print measured GPU time measured with CUDA events
    float msGPU;
    cudaEventElapsedTime(&msGPU, start, end);
    std::cout << "Measured time (CUDA events): " << msGPU << "ms\n";

    /*
    The difference between the two methods, CPU timing and events, is
    important when writing more complex projects: kernels are being 
    launched asynchronously. The launch returns immediately so the CPU
    can progress with other jobs. This means that to get a proper timing,
    we always have to synchronize CPU and GPU before measuring current time
    with chrono. With CUDA events, we can insert them into streams before
    and after the actions we want to measure. We can have multiple events
    inserted at many different points. We still have to synchronize, but 
    only when we eventually want to ACCESS the measurements on the CPU 
    (e.g., once for all timings at the end of a frame to get a report).

    Make sure that you don't try to measure parts of your program with 
    events that mix GPU and CPU code. Events for start and end should 
    only enclose code portions with GPU tasks. Otherwise you won't be 
    sure what you are measuring and might get non-reproducible results!
    */

    //Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    /*
    Dependencies across streams:

    Events may also be used to introduce dependencies
    across streams. One stream may compute an important
    piece of information that another should use. This
    dependency can be modelled by recording an event in
    one stream and have the target stream wait on this 
    event. Commands launched to the stream will not 
    continue until the event is observed.
    */

    // Create a new event to signal that data is ready
    cudaEvent_t fooReady;
    cudaEventCreate(&fooReady);

    // Create two streams, one producer, one consumer
    cudaStream_t producer, consumer;
    cudaStreamCreate(&producer);
    cudaStreamCreate(&consumer);

    /* 
    Enforce the following behavior for producer/consumer streams:

        Producer    Consumer
           |            .
      slow kernel       .
           |            .
       sets foo         .
           \____________.
                        |
                    print foo
    */

    // Producer stream simulates some hard work
    SlowKernel<<<1, 1, 0, producer>>>();
    // Producer sets foo to an important value
    SetFoo<<<1, 1, 0, producer>>>(42);
    // Producer notifies consumer stream that foo is ready
    cudaEventRecord(fooReady, producer);

    // Consumer waits for ready event
    cudaStreamWaitEvent(consumer, fooReady);
    // Without waiting, consumer MAY print before foo is ready!
    PrintFoo<<<1, 1, 0, consumer>>>();

    // Wait for printf outputs
    cudaDeviceSynchronize();
 
    return 0;
}
