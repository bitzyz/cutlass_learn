#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
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

    int num = 0;
    float error;
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

template <typename T, int BM, int BN, int BK, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB,
          typename SmemLayoutA, typename SmemLayoutB,
          typename S2RCopyAtomA, typename S2RCopyAtomB>
__global__ void gemm_shm_v2(const T *Aptr, const T *Bptr, T *Dptr, int m, int n, int k)
{
    // Initilize shared memory
    extern __shared__ T shm_data[];

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    // Initilize thread block
    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

    // Global Memory
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));  // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));  // (BN, BK, num_tile_k)
    Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix)); // (BM, BN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
                          SmemLayoutA{});                      // (BM, BK)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK)

    // register, use tiled_mma to partition register A/B/C
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
    clear(tCrD);

    // from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    //   if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //   {

    //     PRINT("tCrA", tCrA.shape())
    //     PRINT("tCrB", tCrB.shape())

    //       PRINT("tAgA_copy", tAgA_copy.shape())
    //       PRINT("tAsA_copy", tAsA_copy.shape())
    //       // print(layout<0>(tAgA));
    //       // PRINT("tArA", tArA.shape())
    //       PRINT("tBgB_copy", tBgB_copy.shape())
    //       PRINT("tBsB_copy", tBsB_copy.shape())

    //       PRINT("tAsA", tAsA.shape())
    //       PRINT("tCrA_view", tCrA_view.shape())
    //       // print(layout<0>(tBgB));
    //       // PRINT("tBrB", tBrB.shape())

    //       PRINT("tBsB", tBsB.shape())
    //       PRINT("tCrB_view", tCrB_view.shape())
    //   }

    // loop over k: i. load tile, ii. mma
    int ntile = k / BK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile)
    {
        // copy  (CPY, CPY_M, CPY_K) , async
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile),
                   tAsA_copy(_, _, _));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile),
                   tBsB_copy(_, _, _));
        cp_async_fence();

        cp_async_wait<0>();
        __syncthreads();

        int nk = size<2>(tCrA);
#pragma unroll
        for (int ik = 0; ik < nk; ++ik)
        {
            // copy  (CPY, CPY_M), sync
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik),
                       tCrA_view(_, _, ik));
            // copy  (CPY, CPY_N)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik),
                       tCrB_view(_, _, ik));
            // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik
    }     // itile

    // register to global memory
    cute::copy(tCrD, tCgD);
}

template <typename T>
void gemm_v2(T *a, T *b, T *c, int M, int N, int K)
{

    auto BM = Int<32>{};
    auto BN = Int<256>{};
    auto BK = Int<256>{};
    // Define the smem layouts
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<BK>{}),
                    make_stride(Int<BK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BM>{}, Int<BK>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<BN>{}, Int<BK>{}))); // (m,n) -> smem_idx

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 2;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 2 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
    using G2SCopyB = G2SCopyA;

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    dim3 block(size(MMA{}));
    dim3 grid(BX, BY);

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize =
        shm_size_AB * sizeof(T);

    int shm_size = kShmSize;

    cudaFuncSetAttribute(gemm_shm_v2<T, BM, BN, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, S2RCopyAtomA, S2RCopyAtomB>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    gemm_shm_v2<T, BM, BN, BK, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB, S2RCopyAtomA, S2RCopyAtomB>
        <<<grid, block, shm_size>>>(a, b, c, M, N, K);
}

int main()
{
    const int repeat = 100;

    printf("\nalgo = Cute_HGEMM_V2\n");

    // const int M = 256, N = 256, K = 256;
    const int M = 32, N = 10240, K = 8192;
    testF16F16GemmMaxError<T>(
        gemm_v2, compare, M, N, K, repeat);

    // testF16F16GemmPerformance<T>(
    //     gemm_v1, M, N, K, repeat);

    return 0;
}