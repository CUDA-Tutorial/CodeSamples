#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <thread>

// A kernel that wastes some time
__global__ void SlowKernel()
{
    const int start = clock();
    while ((clock() - start) < 100'000'000);
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
    std::cout << "==== Sample 09 - Events ====\n" << std::endl;
    /*
     Using events to measure time and communicate across streams.

     Expected output: 
     1) Longer duration measured with chrono, since it includes 
     CPU-side work, which is not captured by CUDA events.
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
    // Launch a light-weight GPU kernel
    SetFoo<<<1,1>>>(1);
    // Simulate some heavy CPU work
    std::this_thread::sleep_for(2s);
    // Launch another light-weight GPU kernel
    SetFoo<<<1,1>>>(0);
    // Record end directly after last relevant GPU command
    cudaEventRecord(end);
    /*
    Synchronize, for two different reasons: first, to get 
    an adequate time measurement on the CPU after the GPU
    has finished running. Second, to make sure the end 
    event has taken place when we call cudaEventElapsedTime
    below. If we only needed to wait for the event, we 
    could use cudaEventSynchronize instead.
    */
    cudaDeviceSynchronize();
    auto after = std::chrono::system_clock::now();

    // Print measured CPU time - includes work done by the CPU inbetween
    float msCPU = 1000.f * duration_cast<duration<float>>(after - before).count();
    std::cout << "Measured time (chrono): " << msCPU << "ms\n";
    
    // Print measured GPU time - duration of GPU tasks only
    float msGPU;
    cudaEventElapsedTime(&msGPU, start, end);
    std::cout << "Measured time (CUDA events): " << msGPU << "ms\n";

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
