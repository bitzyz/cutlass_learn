#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <float.h>
#include "utils.h"

using T = cute::half_t;
using namespace cute;

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n)
    {
        return;
    }

    float v0 = x[idx];
    float v1 = y[idx];

    float diff = fabs(v0 - v1);
    if (diff > threshold)
    {
        atomicAdd(count, 1);

        // for positive floating point, there int representation is in the same
        // order.
        int int_diff = *((int *)(&diff));
        atomicMax((int *)max_error, int_diff);
    }
}

template <typename T>
void compare(const T *x, const T *y, int n, float threshold)
{
    int *num_count;
    float *max_error;
    cudaMalloc(&num_count, sizeof(int));
    cudaMalloc(&max_error, sizeof(float));
    cudaMemset(num_count, 0, sizeof(int));
    cudaMemset(max_error, 0, sizeof(float));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
    cudaDeviceSynchronize();

    int num = 0;
    float error = 0;
    cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (num == 0)
    {
        printf("check ok, max_error = %f\n", error);
    }
    else
    {
        float p = (100.f * num) / n;
        printf("===============================\n");
        printf("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
               error);
        printf("===============================\n");
    }
}

template <typename T, int BM, int BN, int BK, typename TiledMMA>
__global__ void gemm_cute_v1(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k)
{

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));  // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));  // (BN, BK, num_tile_k)
    Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); // (BM, BN)

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {

    //     PRINT("gA", gA.shape())
    //     PRINT("tAgA", tAgA.shape())
    //     print(layout<0>(tAgA));
    //     PRINT("tArA", tArA.shape())

    //     PRINT("gB", gB.shape())
    //     PRINT("tBgB", tBgB.shape())
    //     print(layout<0>(tBgB));
    //     PRINT("tBrB", tBrB.shape())

    //     PRINT("gC", gC.shape())
    //     PRINT("tCgC", tCgC.shape())
    //     print(layout<0>(tCgC));
    //     PRINT("tCrC", tCrC.shape())
    // }

    clear(tCrC);

    int num_tile_k = size<2>(gA);
#pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile)
    {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}

template <typename T>
void gemm_v1(T *a, T *b, T *c, int m, int n, int k)
{

    using mma_op = SM80_16x8x16_F16F16F16F16_TN; // A 行优先 B 列优先
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // using MMA = decltype(make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
    //                                         Layout<Shape<_2, _4>,
    //                                         Stride<_4,_1>>{})
    //                                     );

    using MMA = decltype(make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                        Layout<Shape<_1, _1, _1>>{})); // Tiler

    // TiledMMA mma1 = make_tiled_mma(mma_atom{},
    //                             Layout<Shape<_1,_1,_1>,S>{});

    // print_latex(mma1);

    // constexpr int BM = 128;
    constexpr int BM = 32;
    constexpr int BN = 256;
    constexpr int BK = 32;

    int BX = (n + BN - 1) / BN;
    int BY = (m + BM - 1) / BM;

    // print(size(MMA{}));

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY);

    gemm_cute_v1<T, BM, BN, BK, MMA><<<grid, block>>>(a, b, c, m, n, k);
}

int main()
{
    const int repeat = 100;

    printf("\nalgo = Cute_HGEMM_V1\n");

    // const int M = 256, N = 256, K = 256;
    const int M = 32, N = 10240, K = 1024;
    testF16F16GemmMaxError<T>(
        gemm_v1, compare, M, N, K, repeat);

    // testF16F16GemmPerformance<T>(
    //     gemm_v1, M, N, K, repeat);

    return 0;
}
