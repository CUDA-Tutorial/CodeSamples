// Define an unsigned integer variable that the GPU can work with
__device__ unsigned int step = 0;

// Increment the GPU variable N times. Whenever a thread observes
// non-consecutive numbers, it prints the latest sequence. Hence,
// every thread documents the turns that it was given by the 
// scheduler. 
__device__ void takeNTurns(const char* who, unsigned int N)
{
    unsigned int lastTurn = 0xFFFFFFFEU, turn, start;
    for (int i = 0; i < N; i++)
    {
        turn = atomicInc(&step, 0xFFFFFFFFU);

        const bool reset = (lastTurn != (turn - 1));
        const bool end = (i == (N - 1));

        if (((i > 0) && reset) || end)
            printf("%s: %03d--%03d\n", who, start, lastTurn + end);

        lastTurn = turn;

        if (reset)
            start = turn;
    }
}

// A simple kernel with two nested if clauses, 4 branches.
// Each thread will take a separate branch, and then perform
// N stepts. With legacy scheduling, each branch must be 
// finished before execution can continue with the next.
__global__ void testScheduling(int N)
{
    if (threadIdx.x < 2) // Branch once
        if (threadIdx.x < 1) // Branch again
            takeNTurns("Thread 1", N);
        else
            takeNTurns("Thread 2", N);
    else
        if (threadIdx.x < 3) // Branch again
            takeNTurns("Thread 3", N);
        else
            takeNTurns("Thread 4", N);
}

void run2NestedBranchesForNSteps(int N)
{
	testScheduling<<<1, 4>>>(N);
	cudaDeviceSynchronize();
}