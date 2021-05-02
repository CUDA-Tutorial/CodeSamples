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

// Matrix multiplication kernel called by MatMul()
template <unsigned int DIM>
__global__ void MatMulKernelTensor(half* A, half* B, float* C)
{
    using namespace nvcuda;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
   
    // Initialize the output
    wmma::fill_fragment(acc_frag, 0.0f);

    // Tile using a 1D grid
    int warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int aRow = 16 * (warpID / (DIM / 16));
    int bCol = 16 * (warpID % (DIM / 16));

    // Loop over the K-dimension, assuming square matrix
    for (int i = 0; i < DIM; i += 16) 
    {
        // Bounds checking
        if (aRow < DIM && bCol < DIM) 
        {    
            //Load the inputs
            wmma::load_matrix_sync(a_frag, A + i + aRow * DIM, DIM);
            wmma::load_matrix_sync(b_frag, B + bCol + i * DIM, DIM);
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    wmma::store_matrix_sync(C + bCol + aRow * DIM, acc_frag, DIM, wmma::mem_row_major);
}

int main()
{
    std::cout << "==== Sample 16 - Tensor Cores ====\n" << std::endl;

    constexpr unsigned int DIM = 512;

    std::cout << "Multiplying two " << DIM << " x " << DIM << " matrices on GPU\n" << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<half> A(DIM * DIM), B(DIM * DIM);
    std::vector<float> C(DIM * DIM);

    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> dist;
    for (int i = 0; i < DIM * DIM; i++)
    {
        A[i] = dist(eng);
        B[i] = dist(eng);
    }

    half* dAPtr, * dBPtr;
    float* dCPtr;
    cudaMalloc(&dAPtr, DIM * DIM * sizeof(half));
    cudaMalloc(&dBPtr, DIM * DIM * sizeof(half));
    cudaMalloc(&dCPtr, DIM * DIM * sizeof(float));
    cudaMemcpy(dAPtr, A.data(), sizeof(half) * DIM * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(dBPtr, B.data(), sizeof(half) * DIM * DIM, cudaMemcpyHostToDevice);

    int device;
    cudaGetDevice(&device);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    enum class METHOD { REF, TENSOR, CUBLAS_NO_TENSOR, CUBLAS};

    for (METHOD m : {METHOD::REF, METHOD::TENSOR, METHOD::CUBLAS_NO_TENSOR, METHOD::CUBLAS})
    {
        cudaMemset(dCPtr, 0, sizeof(float) * DIM * DIM);

        cudaEventRecord(start);
        if (m == METHOD::REF)
        {
            dim3 grid(DIM / 16, DIM / 16);
            dim3 block(16, 16);

            samplesutil::MatMulKernel<DIM, 16><<<grid, block>>>(dAPtr, dBPtr, dCPtr);
        }
        else if (m == METHOD::TENSOR)
        {
            unsigned int warps_required = (DIM * DIM) / (16 * 16);
            unsigned int warps_per_block = 256 / 32;
            unsigned int blocks_required = warps_required / warps_per_block;

            MatMulKernelTensor<DIM><<<blocks_required, 256>>>(dAPtr, dBPtr, dCPtr);
        }
        else if (m == METHOD::CUBLAS_NO_TENSOR || m == METHOD::CUBLAS)
        {
            if(m == METHOD::CUBLAS_NO_TENSOR)
                cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            else
                cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);                

            float alpha = 1.f, beta = 1.f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, DIM, DIM, DIM, &alpha,
                dAPtr, CUDA_R_16F, DIM, dBPtr, CUDA_R_16F, DIM, &beta, 
                dCPtr, CUDA_R_32F, DIM, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms;
        cudaEventElapsedTime(&ms, start, end);
        std::cout << ms << "ms\n" << std::endl;
    }

    cublasDestroy(handle);

    return 0;
}