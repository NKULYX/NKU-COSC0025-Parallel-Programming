#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h> // pthread
#include <semaphore.h>
#include <omp.h>
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
const int LOOP = 50;
float **data;
float **matrix;

ofstream res_stream;

void init_data(int);
void init_matrix();
void calculate_serial();
void calculate_SIMD();
void calculate_openmp_single_simd();
void calculate_pthread();
void calculate_openmp_schedule_static();
void calculate_openmp_schedule_dynamic();
void calculate_openmp_schedule_guided();
void calculate_openmp_schedule_guided_nowait();
void calculate_openmp_schedule_guided_SIMD();
void calculate_openmp_static_thread();
void calculate_openmp_dynamic_thread();
void calculate_openmp_row();
void calculate_openmp_column();
void print_matrix();
void test(int);

int main()
{
    res_stream.open("result.csv", ios::out);
    for (int i = 100; i <= 1000; i += 100)
        test(i);
    for (int i = 1000; i <= 3000; i += 500)
        test(i);
    res_stream.close();
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

// SIMD并行算法
void calculate_SIMD()
{
    for (int k = 0; k < N; k++)
    {
        float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t Akj = vld1q_f32(matrix[k] + j);
            Akj = vdivq_f32(Akj, Akk);
            vst1q_f32(matrix[k] + j, Akj);
        }
        for (; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
            }
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 单独使用openmp进行simd优化
void calculate_openmp_single_simd()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(1) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
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
            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
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
                int j = k + 1;
                float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
                for (j = k + 1; j + 3 < N; j += 4)
                {
                    float32x4_t Akj = vld1q_f32(matrix[k] + j);
                    float32x4_t Aij = vld1q_f32(matrix[i] + j);
                    float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                    Aij = vsubq_f32(Aij, AikMulAkj);
                    vst1q_f32(matrix[i] + j, Aij);
                }
                for (; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }

        // 所有线程进入下一轮
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

// pthread 并行算法
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

// 静态数据划分
void calculate_openmp_schedule_static()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : static)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 动态数据划分
void calculate_openmp_schedule_dynamic()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : dynamic)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 自动调整数据划分
void calculate_openmp_schedule_guided()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 使用nowait
void calculate_openmp_schedule_guided_nowait()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided) nowait
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// openmp + simd
void calculate_openmp_schedule_guided_SIMD()
{
    int i, j, k;
    float tmp;
    float32x4_t Akk, Akj, Aik, Aij, AikMulAkj;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp, Akk, Akj, Aik, Aij, AikMulAkj) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
        // float Akk = matrix[k][k];
        Akk = vmovq_n_f32(matrix[k][k]);
        int j;
        // 并行处理
        tmp = matrix[k][k];
#pragma omp single
        {
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = vld1q_f32(matrix[k] + j);
                // Akj = Akj / Akk;
                Akj = vdivq_f32(Akj, Akk);
                // Akj = matrix[k][j];
                vst1q_f32(matrix[k] + j, Akj);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (int i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
            // float Aik = matrix[i][k];
            Aik = vmovq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4)
            {
                // float Akj = matrix[k][j];
                Akj = vld1q_f32(matrix[k] + j);
                // float Aij = matrix[i][j];
                Aij = vld1q_f32(matrix[i] + j);
                // AikMulAkj = matrix[i][k] * matrix[k][j];
                AikMulAkj = vmulq_f32(Aik, Akj);
                // Aij = Aij - AikMulAkj;
                Aij = vsubq_f32(Aij, AikMulAkj);
                // matrix[i][j] = Aij;
                vst1q_f32(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 静态线程版本
void calculate_openmp_static_thread()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 动态线程版本
void calculate_openmp_dynamic_thread()
{
    int i, j, k;
    float tmp;
    for (k = 0; k < N; k++)
    {
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, tmp) shared(k, matrix, N)
        {
#pragma omp single
            {
                tmp = matrix[k][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[k][j] = matrix[k][j] / tmp;
                }
                matrix[k][k] = 1.0;
            }
#pragma omp for schedule(simd \
                         : guided)
            for (i = k + 1; i < N; i++)
            {
                tmp = matrix[i][k];
#pragma omp simd
                for (j = k + 1; j < N; j++)
                {
                    matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
                }
                matrix[i][k] = 0;
            }
        }
    }
}

// 按行划分
void calculate_openmp_row()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k, tmp) shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = matrix[k][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[k][j] = matrix[k][j] / tmp;
            }
            matrix[k][k] = 1.0;
        }
#pragma omp for schedule(simd \
                         : guided)
        for (i = k + 1; i < N; i++)
        {
            tmp = matrix[i][k];
#pragma omp simd
            for (j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - tmp * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

// 按列划分
void calculate_openmp_column()
{
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(matrix, N)
    for (k = 0; k < N; k++)
    {
#pragma omp for schedule(simd \
                         : guided)
        for (j = k + 1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (i = k + 1; i < N; i++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
        }
#pragma omp single
        {
            matrix[k][k] = 1;
            for (i = k + 1; i < N; i++)
            {
                matrix[i][k] = 0;
            }
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

void test(int n)
{
    N = n;
    cout << "=================================== " << N << " ===================================" << endl;
    res_stream << N;
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
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== SIMD ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "SIMD:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_single_SIMD ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_single_simd();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_single_SIMD:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
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
    res_stream << "," << time / LOOP;
    //    print_matrix();
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
    res_stream << "," << time / LOOP;
    //    print_matrix();
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
    res_stream << "," << time / LOOP;
    //    print_matrix();
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
    cout << "openmp_schedule_guided:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_schedule_guided_nowait ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_guided_nowait();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_schedule_guided_nowait:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_schedule_guided_SIMD ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_schedule_guided_SIMD();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_schedule_guided_SIMD:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_static_thread ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_static_thread();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_static_thread:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_dynamic_thread ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_dynamic_thread();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_dynamic_thread:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_row ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_row();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_row:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    //    print_matrix();
    // ====================================== openmp_column ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_openmp_column();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "openmp_column:" << time / LOOP << "ms" << endl;
    res_stream << "," << time / LOOP;
    res_stream << endl;
    //    print_matrix();
}