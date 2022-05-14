#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <xmmintrin.h> // SSE
#include <pthread.h>   // pthread
#include <semaphore.h>
#include <omp.h>

#define _PRINT
// #define _TEST

using namespace std;

int NUM_THREADS = 8;
// ============================================== pthread 线程控制变量 ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
// ============================================== 运算变量 ==============================================
int N;
const int L = 100;
const int LOOP = 1;
float **data;
float **matrix;

ofstream res_stream;

void init_data();
void init_matrix();
void calculate_serial();
void calculate_SIMD();
void calculate_openmp_offloading(float*);
void print_matrix();
void test(int);
void print_result(int);

int main()
{
    #ifdef _TEST
    res_stream.open("result.csv", ios::out);
    for (int i = 100; i <= 1000; i += 100)
        test(i);
    for (int i = 1000; i <= 3000; i += 500)
        test(i);
    res_stream.close();
    #endif
    #ifdef _PRINT
        test(10);
    #endif
    system("pause");
    return 0;
}

void init_data()
{
    data = new float *[N], matrix = new float *[N];
    for (int i = 0; i < N; i++)
        data[i] = new float[N], matrix[i] = new float[N];
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            data[i][j] = rand() * 1.0 / RAND_MAX * L;
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                data[j][k] += data[i][k];
}

// 用data初始化matrix，保证每次进行计算的数据是一致的
void init_matrix()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = data[i][j];
}

// 串行算法
void calculate_serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// openmp offloading
void calculate_openmp_offloading(float * buffer)
{
    int is_cpu = true;
    float * buf = buffer;
#pragma omp target map(tofrom: buf[0:N*N]) map(from: is_cpu) map(to: N)
    {
        int i, j, k;
        is_cpu = omp_is_initial_device();

        for (k = 0; k < N; k++) {
#pragma omp parallel default(none), private(i, j), shared(buf, N, k)
            {
#pragma omp single
                {
                    for (j = k + 1; j < N; j++) {
                        buf[k*N+j] = buf[k*N+j] / buf[k*N+k];
                    }
                    buf[k*N+k] = 1;
                }
#pragma omp for simd
                for (i = k + 1; i < N; i++) {
                    for (j = k + 1; j < N; j++) {
                        buf[i*N+j] = buf[i*N+j] - buf[i*N+k] * buf[k*N+j];
                    }
                    buf[i*N+k] = 0;
                }
            }
        }
    }
    cout<<(is_cpu? "CPU":"GPU")<<endl;
}

// 打印矩阵
void print_matrix()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void test(int n)
{
    N = n;
    cout << "=================================== " << N << " ===================================" << endl;
    #ifdef _TEST
    res_stream << N;
    #endif
    struct timeval start;
    struct timeval end;
    float time = 0;
    init_data();
    // ====================================== serial ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_serial();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "serial:" << time / LOOP << "ms" << endl;
    print_result(time);
    // ====================================== openmp offloading ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        float * buffer = new float[N*N];
        for(int i = 0; i < N*N; i++)
            buffer[i] = matrix[i/N][i%N];
        gettimeofday(&start, NULL);
        calculate_openmp_offloading(buffer);
        gettimeofday(&end, NULL);
        // 将buffer复制回matrix
        for(int i = 0; i < N*N; i++)
            matrix[i/N][i%N] = buffer[i];
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_offloading:" << time / LOOP << "ms" << endl;
    print_result(time);
    #ifdef _TEST
    res_stream << endl;
    #endif
}

void print_result(int time)
{
    #ifdef _TEST
    res_stream << "," << time / LOOP;
    #endif
    #ifdef _PRINT
    print_matrix();
    #endif
}