#include <stdio.h>

void initWith(float num, float *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
    int begin = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = gridDim.x * blockDim.x;
    for (int i = begin; i < N; i += gridStride)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2 << 20;
    size_t size = N * sizeof(float);

    cudaError_t err_tmp;
    cudaError_t err = cudaSuccess;

    float *a;
    float *b;
    float *c;

    err_tmp = cudaMallocManaged(&a, size);
    err = err_tmp == cudaSuccess ? err : err_tmp;
    err_tmp = cudaMallocManaged(&b, size);
    err = err_tmp == cudaSuccess ? err : err_tmp;
    err_tmp = cudaMallocManaged(&c, size);
    err = err_tmp == cudaSuccess ? err : err_tmp;

    if (err != cudaSuccess)
    {
        printf("cuda memory error occur : %s\n", cudaGetErrorString(err));
    }

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    addVectorsInto<<<2, 5>>>(c, a, b, N);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cuda kernel function error occur : %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cuda synchronize error occur : %s\n", cudaGetErrorString(err));
    }

    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
