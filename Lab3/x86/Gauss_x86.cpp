#include <iostream>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512
using namespace std;

//------------------------------------------ 线程控制变量 ------------------------------------------
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;

const int THREAD_NUM = 8;

// ------------------------------------------ 全局计算变量 ------------------------------------------
const int N = 20;
const int L = 100;
const int LOOP = 1;
float data[N][N];
float matrix[N][N];

void init_data();
void init_matrix();
void calculate_serial();
void calculate_SSE();
void calculate_pthread_SSE();
void calculate_AVX();
void calculate_pthread_AVX();
void calculate_AVX512();
void calculate_pthread_AVX512();
void print_matrix();

int main()
{
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
    // ====================================== SSE ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_SSE();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "SSE:" << time / LOOP << "ms" << endl;
    // ====================================== pthread_SSE ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread_SSE();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread_SSE:" << time / LOOP << "ms" << endl;
    // ====================================== AVX ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_AVX();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "AVX:" << time / LOOP << "ms" << endl;
    // ====================================== pthread_AVX ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread_AVX();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread_AVX:" << time / LOOP << "ms" << endl;
    // ====================================== AVX512 ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_AVX512();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "AVX512:" << time / LOOP << "ms" << endl;
    // ====================================== pthread_AVX512 ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread_AVX512();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread_AVX512:" << time / LOOP << "ms" << endl;
    system("pause");
}

// 初始化data，保证每次数据都是一致的
void init_data()
{
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

// SSE并行算法
void calculate_SSE()
{
    for (int k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        __m128 Akk = _mm_set_ps1(matrix[k][k]);
        int j;
        // 并行处理
        for (j = k + 1; j + 3 < N; j += 4)
        {
            // float Akj = matrix[k][j];
            __m128 Akj = _mm_loadu_ps(matrix[k] + j);
            // Akj = Akj / Akk;
            Akj = _mm_div_ps(Akj, Akk);
            // Akj = matrix[k][j];
            _mm_storeu_ps(matrix[k] + j, Akj);
        }
        // 串行处理结尾
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // float Aik = matrix[i][k];
            __m128 Aik = _mm_set_ps1(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm_storeu_ps(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// pthread_SSE 线程函数
void *threadFunc_SSE(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m128 Akk = _mm_set_ps1(matrix[k][k]);
            int j;
            //考虑对齐操作
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < N; i += (THREAD_NUM - 1))
            {
                // float Aik = matrix[i][k];
                __m128 Aik = _mm_set_ps1(matrix[i][k]);
                int j = k + 1;
                for (; j + 3 < N; j += 4)
                {
                    // float Akj = matrix[k][j];
                    __m128 Akj = _mm_loadu_ps(matrix[k] + j);
                    // float Aij = matrix[i][j];
                    __m128 Aij = _mm_loadu_ps(matrix[i] + j);
                    // AikMulAkj = matrix[i][k] * matrix[k][j];
                    __m128 AikMulAkj = _mm_mul_ps(Aik, Akj);
                    // Aij = Aij - AikMulAkj;
                    Aij = _mm_sub_ps(Aij, AikMulAkj);
                    // matrix[i][j] = Aij;
                    _mm_storeu_ps(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_SSE 并行算法
void calculate_pthread_SSE()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_SSE, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// AVX 并行算法
void calculate_AVX()
{
    for (int k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        __m256 Akk = _mm256_set1_ps(matrix[k][k]);
        int j;
        // 并行处理
        for (j = k + 1; j + 7 < N; j += 8)
        {
            // float Akj = matrix[k][j];
            __m256 Akj = _mm256_loadu_ps(matrix[k] + j);
            // Akj = Akj / Akk;
            Akj = _mm256_div_ps(Akj, Akk);
            // Akj = matrix[k][j];
            _mm256_storeu_ps(matrix[k] + j, Akj);
        }
        // 串行处理结尾
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // float Aik = matrix[i][k];
            __m256 Aik = _mm256_set1_ps(matrix[i][k]);
            for (j = k + 1; j + 7 < N; j += 8)
            {
                // float Akj = matrix[k][j];
                __m256 Akj = _mm256_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                __m256 Aij = _mm256_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                __m256 AikMulAkj = _mm256_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm256_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm256_storeu_ps(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// pthread_AVX 线程函数
void *threadFunc_AVX(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m256 Akk = _mm256_set1_ps(matrix[k][k]);
            int j;
            //考虑对齐操作
            for (j = k + 1; j + 7 < N; j += 8)
            {
                // float Akj = matrix[k][j];
                __m256 Akj = _mm256_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm256_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm256_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < N; i += (THREAD_NUM - 1))
            {
                // float Aik = matrix[i][k];
                __m256 Aik = _mm256_set1_ps(matrix[i][k]);
                int j = k + 1;
                for (; j + 7 < N; j += 8)
                {
                    // float Akj = matrix[k][j];
                    __m256 Akj = _mm256_loadu_ps(matrix[k] + j);
                    // float Aij = matrix[i][j];
                    __m256 Aij = _mm256_loadu_ps(matrix[i] + j);
                    // AikMulAkj = matrix[i][k] * matrix[k][j];
                    __m256 AikMulAkj = _mm256_mul_ps(Aik, Akj);
                    // Aij = Aij - AikMulAkj;
                    Aij = _mm256_sub_ps(Aij, AikMulAkj);
                    // matrix[i][j] = Aij;
                    _mm256_storeu_ps(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_AVX 并行算法
void calculate_pthread_AVX()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_AVX, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

// AVX512 并行算法
void calculate_AVX512()
{
    for (int k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        __m512 Akk = _mm512_set1_ps(matrix[k][k]);
        int j;
        // 并行处理
        for (j = k + 1; j + 15 < N; j += 16)
        {
            // float Akj = matrix[k][j];
            __m512 Akj = _mm512_loadu_ps(matrix[k] + j);
            // Akj = Akj / Akk;
            Akj = _mm512_div_ps(Akj, Akk);
            // Akj = matrix[k][j];
            _mm512_storeu_ps(matrix[k] + j, Akj);
        }
        // 串行处理结尾
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // float Aik = matrix[i][k];
            __m512 Aik = _mm512_set1_ps(matrix[i][k]);
            for (j = k + 1; j + 15 < N; j += 16)
            {
                // float Akj = matrix[k][j];
                __m512 Akj = _mm512_loadu_ps(matrix[k] + j);
                // float Aij = matrix[i][j];
                __m512 Aij = _mm512_loadu_ps(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                __m512 AikMulAkj = _mm512_mul_ps(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = _mm512_sub_ps(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                _mm512_storeu_ps(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// AVX512 线程函数
void *threadFunc_AVX512(void *param)
{
    threadParam_t *thread_param_t = (threadParam_t *)param;
    int t_id = thread_param_t->t_id;
    for (int k = 0; k < N; k++)
    {
        // 如果当前是0号线程，则进行除法操作，其余线程处于等待状态
        if (t_id == 0)
        {
            // float Akk = matrix[k][k];
            __m512 Akk = _mm512_set1_ps(matrix[k][k]);
            int j;
            //考虑对齐操作
            for (j = k + 1; j + 15 < N; j += 16)
            {
                // float Akj = matrix[k][j];
                __m512 Akj = _mm512_loadu_ps(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = _mm512_div_ps(Akj, Akk);
                // Akj = matrix[k][j];
                _mm512_storeu_ps(matrix[k] + j, Akj);
            }
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Division);
        }

        // 除法操作完成后，如果是0号线程，则需要唤醒其他线程
        if (t_id == 0)
        {
            for (int i = 1; i < THREAD_NUM; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < N; i += (THREAD_NUM - 1))
            {
                // float Aik = matrix[i][k];
                __m512 Aik = _mm512_set1_ps(matrix[i][k]);
                int j = k + 1;
                for (; j + 15 < N; j += 16)
                {
                    // float Akj = matrix[k][j];
                    __m512 Akj = _mm512_loadu_ps(matrix[k] + j);
                    // float Aij = matrix[i][j];
                    __m512 Aij = _mm512_loadu_ps(matrix[i] + j);
                    // AikMulAkj = matrix[i][k] * matrix[k][j];
                    __m512 AikMulAkj = _mm512_mul_ps(Aik, Akj);
                    // Aij = Aij - AikMulAkj;
                    Aij = _mm512_sub_ps(Aij, AikMulAkj);
                    // matrix[i][j] = Aij;
                    _mm512_storeu_ps(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // 所有线程准备进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread_AVX512 并行算法
void calculate_pthread_AVX512()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc_AVX512, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < THREAD_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
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