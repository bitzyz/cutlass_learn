#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <chrono>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define OFFSETCOL(row, col, ld) ((col) * (ld) + (row))

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINTTENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");

template <typename T>
void testF16F16GemmMaxError(
    void (*gpuF16F16Gemm)(T *, T *, T *, int, int, int),
    void (*compare)(const T *, const T *, int, float),
    int M, int N, int K, int repeat)
{

    size_t size_a = M * K * sizeof(T);
    size_t size_b = K * N * sizeof(T);
    size_t size_c = M * N * sizeof(T);

    T *Aptr_host, *Bptr_host, *Cptr_host;
    T *Aptr_device, *Bptr_device, *Cptr_device;

    Aptr_host = (T *)malloc(size_a);
    Bptr_host = (T *)malloc(size_b);
    Cptr_host = (T *)malloc(size_c);
    cudaMalloc(&Aptr_device, size_a);
    cudaMalloc(&Bptr_device, size_b);
    cudaMalloc(&Cptr_device, size_c);
    cudaMemset(Cptr_device, 0, size_c);

    srand(2333);
    for (int i = 0; i < M * K; i++)
        // Aptr_host[i] = (T)(((rand() % 200) - 100.f) * 0.01f);
        Aptr_host[i] = (T)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        Bptr_host[i] = (T)(rand() / float(RAND_MAX));

    // cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);
    cudaMemcpy(Aptr_device, Aptr_host, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(Bptr_device, Bptr_host, size_b, cudaMemcpyHostToDevice);

    // run cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    int cublas_version;
    cublasGetVersion(handle, &cublas_version);
    printf("cublas version = %d\n", cublas_version);

    T *Cptr_cublas;
    T *Cptr_cublas_host;
    Cptr_cublas_host = (T *)malloc(size_c);
    cudaMalloc(&Cptr_cublas, size_c);
    cudaMemset(Cptr_cublas, 0, size_c);
    half alpha = 1.0f;
    half beta = 0.0f;
    std::chrono::duration<double>
        cublas_totalTime = std::chrono::duration<double>::zero();
    for (int i = 0; i < repeat; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, (half *)Bptr_device, K, (half *)Aptr_device, K, &beta, (half *)Cptr_cublas, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        cublas_totalTime += end - start;
        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
            return;
        }
    }
    float cublas_sec = cublas_totalTime.count() / repeat * 1000.0f; // ms
    printf("cublas time = %f ms\n", cublas_sec);
    cudaMemcpy(Cptr_cublas_host, Cptr_cublas, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // run cutlass

    std::chrono::duration<double> cutlass_totalTime = std::chrono::duration<double>::zero();
    for (int i = 0; i < repeat; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        gpuF16F16Gemm(Aptr_device, Bptr_device, Cptr_device, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        cutlass_totalTime += end - start;
    }
    float cutlass_sec = cutlass_totalTime.count() / repeat * 1000.0f; // ms
    printf("cutlass time = %f ms\n", cutlass_sec);
    // cudaDeviceSynchronize();

    cudaMemcpy(Cptr_host, Cptr_device, size_c, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    compare(Cptr_device, Cptr_cublas, M * N, 0.1);
    cudaDeviceSynchronize();

    free(Aptr_host);
    free(Bptr_host);
    free(Cptr_host);
    free(Cptr_cublas_host);
    cudaFree(Aptr_device);
    cudaFree(Bptr_device);
    cudaFree(Cptr_device);
    cudaFree(Cptr_cublas);

    return;
}
