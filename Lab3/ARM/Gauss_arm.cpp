#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
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
const int N = 10;
const int L = 100;
const int LOOP = 1;
float data[N][N];
float matrix[N][N];

void init_data();
void init_matrix();
void calculate_serial();
void calculate_neon();
void calculate_pthread();
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
    // ====================================== neon ======================================
    time = 0;
    for (int i = 0; i < LOOP; i++)
    {
        init_matrix();
        gettimeofday(&start, NULL);
        calculate_neon();
        gettimeofday(&end, NULL);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    cout << "neon:" << time / LOOP << "ms" << endl;
    print_matrix();
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
    print_matrix();
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

// neon并行算法
void calculate_neon()
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
            for (j = k + 1; j + 3 < n; j += 4)
            {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
            }
            for (; j < n; j++)
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

// pthread并行算法
void calculate_pthread()
{
    // 信号量初始化
    sem_init(&sem_Division, 0, 0);
    pthread_barrier_init(&barrier,NULL,THREAD_NUM);

    // 创建线程
    pthread_t threads[THREAD_NUM];
    threadParam_t thread_param_t[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++)
    {
        thread_param_t[i].t_id = i;
        pthread_create(&threads[i], NULL, threadFunc, (void *)(&thread_param_t[i]));
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
