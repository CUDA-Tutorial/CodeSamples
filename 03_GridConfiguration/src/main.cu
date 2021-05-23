#include <cuda_runtime_api.h>
#include <iostream>

__global__ void PrintIDs()
{
    // Use built-in variables blockIdx and threadIdx
    const auto tID = threadIdx;
    const auto bID = blockIdx;
    printf("Block Id: %d,%d - Thread Id: %d,%d\n", bID.x, bID.y, tID.x, tID.y);
}

int main()
{
    std::cout << "==== Sample 03 - Grid Configurations ====\n" << std::endl;
    /*
    Expected output:
    Block IDs and Thread IDs for two separate grids
    */

    std::cout << "Small grid: \n";
    // Configure the grid and block dimensions via built-in struct dim3 (X,Y,Z)
    const dim3 gridSize_small{ 1, 1, 1 };
    const dim3 blockSize_small{ 4, 4, 1 };

    // Launch kernel with custom grid
    PrintIDs<<<gridSize_small, blockSize_small>>>();

    // Need to synchronize here to have the GPU and CPU printouts in the correct order
    cudaDeviceSynchronize();

    std::cout << "\nLarger grid: \n";
    const dim3 gridSize_large{ 2, 2, 1 };
    const dim3 blockSize_large{ 16, 16, 1 };
    PrintIDs<<<gridSize_large, blockSize_large >>>();
    cudaDeviceSynchronize();

    return 0;
}

/*
Exercises:
1) Launch a 1D grid with 2D blocks (e.g., 4 x (4 x 4))
2) Launch a 2D grid with 2D blocks (e.g., (4 x 4) x (4 x 4))
3) Launch a grid where either the grid or block are 3D
4) Can you find out what the limits are to how many blocks/threads you can launch?
*/