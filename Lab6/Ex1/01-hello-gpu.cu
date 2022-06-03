#include <stdio.h>

void helloCPU()
{
    printf("Hello from the CPU.\n");
}

/*
 * Refactor the `helloGPU` definition to be a kernel
 * that can be launched on the GPU. Update its message
 * to read "Hello from the GPU!"
 */

__global__ void helloGPU()
{
    printf("Hello from the GPU.\n");
}

int main()
{

    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    helloCPU();

    /*
     * Refactor this call to `helloGPU` so that it launches
     * as a kernel on the GPU.
     */

    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    /*
     * Add code below to synchronize on the completion of the
     * `helloGPU` kernel completion before continuing the CPU
     * thread.
     */
}
