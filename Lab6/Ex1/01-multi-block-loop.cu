#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop(int N)
{
    for (int i = 0; i < N; ++i)
    {
        printf("This is iteration number %d in block %d thrad %d\n", threadIdx.x + blockIdx.x * blockDim.x, blockIdx.x, threadIdx.x);
    }
}

int main()
{
    /*
     * When refactoring `loop` to launch as a kernel, be sure
     * to use the execution configuration to control how many
     * "iterations" to perform.
     *
     * For this exercise, be sure to use more than 1 block in
     * the execution configuration.
     */

    int N = 10;
    loop<<<2, 5>>>(N);
    cudaDeviceSynchronize();
}
