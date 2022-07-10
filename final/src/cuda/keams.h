
#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

float** omp_kmeans(int, float**, int, int, int, float, int*);
float** seq_kmeans(float**, int, int, int, float, int*, int*);
float** cuda_kmeans(float**, int, int, int, float, int*, int*);

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);


double  wtime(void);

extern int _debug;

#endif