#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <xmmintrin.h> // SSE
#include <pthread.h> // pthread
#include <semaphore.h>
#include <omp.h>
using namespace std;

int NUM_THREADS = 8;
// ============================================== pthread 线程控制 ==============================================
typedef struct
{
    int t_id;
} threadParam_t;

sem_t sem_Division;
pthread_barrier_t barrier;
// ============================================== 
const int N = 1000;
const int L = 100;
const int LOOP = 1;
float data[N][N];
float matrix[N][N];

void init_data();
void init_matrix();
void calculate_serial();
void calculate_SSE();
void calculate_pthread();
void calculate_openmp_schedule_static();
void calculate_openmp_schedule_dynamic();
void calculate_openmp_schedule_guided();
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
    // ====================================== pthread ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_pthread();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "pthread:" << time / LOOP << "ms" << endl;
    // ====================================== openmp_schedule_static ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_static();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_schedule_static:" << time / LOOP << "ms" << endl;
    // ====================================== openmp_schedule_dynamic ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_dynamic();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_schedule_dynamic:" << time / LOOP << "ms" << endl;
    // ====================================== openmp_schedule_guided ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_guided();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    system("pause");
    return 0;
}

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

// pthread_discrete 线程函数
void *threadFunc(void *param)
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
            for (int i = 1; i < NUM_THREADS; i++)
            {
                sem_post(&sem_Division);
            }
        }
        else
        {
            // 循环划分任务
            for (int i = k + t_id; i < N; i += (NUM_THREADS - 1))
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

// pthread_discrete 并行算法
void calculate_pthread()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    // 创建线程
    pthread_t threads[NUM_THREADS];
    threadParam_t thread_param_t[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc, (void *)(&thread_param_t[i]));
    }

    // 加入执行线程
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 销毁信号量
    sem_destroy(&sem_Division);
    pthread_barrier_destroy(&barrier);
}

void calculate_openmp_schedule_static()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp master
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        #pragma omp for schedule(static)
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



void calculate_openmp_schedule_dynamic()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp master
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        #pragma omp for schedule(dynamic)
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


void calculate_openmp_schedule_guided()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++)
    {
        #pragma omp master
        for (int j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        #pragma omp for schedule(guided)
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