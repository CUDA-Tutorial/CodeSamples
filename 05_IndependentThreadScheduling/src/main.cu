#include <cuda_runtime_api.h>
#include <iostream>
#include "../../shared/include/test_scheduling.cuh"

int main()
{
    std::cout << "==== Sample 05 ====\n";
    std::cout << "==== Independent Thread Scheduling ====\n" << std::endl;
    /*
     This code will launch a particular test kernel.
     It will launch 4 threads in total.
     The program code is structured such that each
     thread enters one of 4 possible branches and then
     atomically increments a GPU variable N times:
    
                .---- N operations by Thread 0
           ----X
         /      '---- N operations by Thread 1
    ----X
         \      .---- N operations by Thread 2
           ----X
                '---- N operations by Thread 3
    
     Each thread will document consecutive ranges of 
     values it observed for the incremented variable.  
     Basically, this will give us an idea how threads
     take turns running in this branching scenario.
    
     Expected output: Many smaller observed value ranges!
     Independent thread scheduling is free to switch 
     between branches to keep the compute units of the 
     GPU busy with work. Thus, threads can alternate 
     taking turns to complete their N steps each.
    
     Disclaimer: behavior is dependent on scheduling at
     runtime. Results may vary.
     */

    const int N = 128;
    run2NestedBranchesForNSteps(N);
    return 0;
}
