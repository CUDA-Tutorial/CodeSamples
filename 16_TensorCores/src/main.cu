#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>
#include <mma.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "../../shared/include/utility.h"

// Matrix multiplication kernel using Tensor Cores
template <unsigned int DIM>
__global__ void MatMulKernelTensor(half* A, half* B, float* C)
{
    // wmma functionality is maintainted in nvcuda namespace
    using namespace nvcuda;

    /*
    Declare the input and output fragments (small matrix tiles). Each fragment
    defines its type (matrix_a and matrix_b are the multiplicands, accumulator
    is one that results are accumulated in), the dimensions NxMxK of the fragment,
    the data type and the layout.
    */
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
   
    // Initialize the output fragment to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    /* 
    Compute row and column indices of the output tile, assuming the kernel
    was launched in a 1D grid with enough warps to process the full matrix.
    */
    int warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int aRow = 16 * (warpID / (DIM / 16));
    int bCol = 16 * (warpID % (DIM / 16));

    // Loop over the K-dimension (assuming square A,B,C matrices)
    for (int i = 0; i < DIM; i += 16) 
    {
        // Bounds checking
        if (aRow < DIM && bCol < DIM) 
        {    
            // Collaboratively load the current tiles (fragments) from pointers A, B
            wmma::load_matrix_sync(a_frag, A + i + aRow * DIM, DIM);
            wmma::load_matrix_sync(b_frag, B + bCol + i * DIM, DIM);
            // Perform the collaborative matrix multiplication on the loaded fragments
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    // When done, write the accumulated matrix back to global memory pointed to by C
    wmma::store_matrix_sync(C + bCol + aRow * DIM, acc_frag, DIM, wmma::mem_row_major);
}

int main()
{
    std::cout << "==== Sample 16 - Tensor Cores ====\n" << std::endl;
    /*
    In this sample, we demonstrate the very common application of 
    matrix-matrix multiplication. To illustrate the benefits of
    tensor cores, we run it in four different ways:
    1) Using tiling and shared memory (CUDA Progrraming Guide)
    2) Using tiling and tensor cores
    3) Using CUBLAS without tensor cores
    4) Using CUBLAS with tensor cores

    Expected output: Slowest times with the approach based on shared
    memory, improvements when using tensor cores. Good performance 
    when using CUBLAS, significatnly better performance with tensor
    cores enabled.
    */

    constexpr unsigned int DIM = 4096;
    std::cout << "Multiplying two " << DIM << " x " << DIM << " matrices on GPU\n" << std::endl;

    // To use CUBLAS functions, we initiate a handle once
    cublasHandle_t handle;
    cublasCreate(&handle);
    /*
    We prepare the memory for data storage in the matrix multiplication.
    We use half types for the multiplicands, and float for the result.
    At the moment, tensor cores cannot support multiplicands of 32-bit 
    floating point types.
    */
    std::vector<half> A(DIM * DIM), B(DIM * DIM);
    std::vector<float> C(DIM * DIM);
    // Fill the multiplicands with random values
    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> dist;
    for (int i = 0; i < DIM * DIM; i++)
    {
        A[i] = dist(eng);
        B[i] = dist(eng);
    }
    // Copy the data to the GPU
    half* dAPtr, * dBPtr;
    float* dCPtr;
    cudaMalloc(&dAPtr, DIM * DIM * sizeof(half));
    cudaMalloc(&dBPtr, DIM * DIM * sizeof(half));
    cudaMalloc(&dCPtr, DIM * DIM * sizeof(float));
    cudaMemcpy(dAPtr, A.data(), sizeof(half) * DIM * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(dBPtr, B.data(), sizeof(half) * DIM * DIM, cudaMemcpyHostToDevice);

    // We create events to perform basic performance measurements
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    /*
     We evaluate four different techniques: a reference implementation from
     the CUDA Programming Guide examples, one naive implementation using
     Tensor Cores, and two CUBLAS methods, one with Tensor Cores diabled,
     the other enabled. 
    */
    enum class METHOD { REF, TENSOR, CUBLAS_NO_TENSOR, CUBLAS};
    for (METHOD m : {METHOD::REF, METHOD::TENSOR, METHOD::CUBLAS_NO_TENSOR, METHOD::CUBLAS})
    {
        // Initiatlize the output matrix
        cudaMemset(dCPtr, 0, sizeof(float) * DIM * DIM);
        // Begin time measurement
        cudaEventRecord(start);
        if (m == METHOD::REF)
        {
            dim3 grid(DIM / 16, DIM / 16);
            dim3 block(16, 16);
            // Reference matrix multiplication, lifted from CUDA Programming Guide
            samplesutil::MatMulKernel<DIM, 16><<<grid, block>>>(dAPtr, dBPtr, dCPtr);
        }
        else if (m == METHOD::TENSOR)
        {
            unsigned int warps_required = (DIM * DIM) / (16 * 16);
            unsigned int warps_per_block = 256 / 32;
            unsigned int blocks_required = warps_required / warps_per_block;
            // Basic tensor cores based implementation of matrix multiplication
            MatMulKernelTensor<DIM><<<blocks_required, 256>>>(dAPtr, dBPtr, dCPtr);
        }
        else if (m == METHOD::CUBLAS_NO_TENSOR || m == METHOD::CUBLAS)
        {
            /*
            By default, tensor cores will be used whenever possible by
            CUBLAS. The only way to disable them still is to use the
            CUBLAS_PEDANTIC_MATH mode.
            */
            if(m == METHOD::CUBLAS_NO_TENSOR)
                cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            else
                cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);                
            /*
            Run CUBLAS GEMM. Note that the output matrices will be
            transposed compared to the other techniques, since CUBLAS
            prefers row-major layout.
            */
            float alpha = 1.f, beta = 1.f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, DIM, DIM, DIM, &alpha,
                dAPtr, CUDA_R_16F, DIM, dBPtr, CUDA_R_16F, DIM, &beta, 
                dCPtr, CUDA_R_32F, DIM, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        // Synchronize and report run time of each individual technique
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        std::cout << ms << "ms\n" << std::endl;
    }
    // Destroy acquired CUBLAS handle
    cublasDestroy(handle);
    return 0;
}

/*
Exercises:
1) Change the tensor matrix multiplication method to work with a different
input data type and adapt input generation accordingly
2) Change the tensor matrix multiplication method to work with a different
output data type and adapt input generation accordingly
3) Change the application to work with ANY matrix size, i.e., one that is
NOT a multiple of 16. You may want to start with a very small size for this.
*/