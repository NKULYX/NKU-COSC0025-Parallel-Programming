#include <iostream>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <windows.h>
using namespace std;

const int N = 16 * 16 * 16;
const int D = 16 * 16 * 16 * 16;
const int K = 4;
const int L = 100;
const int LOOP = 1;

float **data; // 数据集
float **data_align;
float centroids[K][D]; // 聚类中心
int cluster[N];        // 各数据所属类别
int cntCluster[K];     // 个聚类计数

void initData();
void initCentroids();
void calculate_serial();
void calculate_SSE_aligned();
void calculate_SSE_unaligned();
void calculate_AVX_aligned();
void calculate_AVX_unaligned();
void calculate_AVX512_aligned();
void calculate_AVX512_unaligned();
void updateCentroids();

int main()
{
    long long int head, tail, freq;
    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_serial();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_serial: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_SSE_aligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_SSE_aligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_SSE_unaligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_SSE_unaligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_AVX_aligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_AVX_aligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_AVX_unaligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_AVX_unaligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_AVX512_aligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_AVX512_aligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    initData();
    initCentroids();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < LOOP; i++)
    {
        calculate_AVX512_unaligned();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "calculate_AVX512_unaligned: " << (tail - head) * 1000.0 / freq / LOOP << endl;

    system("pause");
}

void initData()
{
    data = new float *[D];
    data_align = (float **)malloc(sizeof(float *) * D);
    for (int i = 0; i < D; i++)
        data[i] = new float[N + 1], data_align[i] = (float *)_aligned_malloc(sizeof(float) * N, 64);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < N; j++)
            data[i][j + 1] = rand() * 1.0 / RAND_MAX * L, data_align[i][j] = rand() * 1.0 / RAND_MAX * L;
}

void initCentroids()
{
    for (int i = 0; i < K; i++)
        for (int j = 0; j < D; j++)
            centroids[i][j] = rand() * 1.0 / RAND_MAX * L;
}

void updateCentroids()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            centroids[cluster[i]][j] += data[j][i];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < D; j++)
            centroids[i][j] /= cntCluster[i];
}

void calculate_serial()
{
    for (int i = 1; i < N + 1; i++)
    {
        float min_dis = L * L;
        for (int j = 0; j < K; j++)
        {
            float dis = 0;
            for (int d = 0; d < D; d++)
                dis += (data[d][i] - centroids[j][d]) * (data[d][i] - centroids[j][d]);
            if (dis < min_dis)
                min_dis = dis, cluster[i] = j, cntCluster[j]++;
        }
    }
}

void calculate_SSE_aligned()
{
    float min_distance[N] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 0; i < N - N % 4; i += 4)
            {
                // 取出原始积累的距离
                __m128 distance = _mm_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[4] = {centroids[j][d]};
                __m128 centroid_d = _mm_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m128 data_d = _mm_load_ps(&data_align[d][i]);
                // 对每一数据该维度计算差值
                __m128 delta = _mm_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm_add_ps(distance, _mm_mul_ps(delta, delta));
                // 存回
                _mm_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 0; i < N; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}

void calculate_SSE_unaligned()
{
    float min_distance[N + 1] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 1; i < N - N % 4 + 1; i += 4)
            {
                // 取出原始积累的距离
                __m128 distance = _mm_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[4] = {centroids[j][d]};
                __m128 centroid_d = _mm_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m128 data_d = _mm_loadu_ps(&data[d][i]);
                // 对每一数据该维度计算差值
                __m128 delta = _mm_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm_add_ps(distance, _mm_mul_ps(delta, delta));
                // 存回
                _mm_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 1; i < N + 1; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}

void calculate_AVX_aligned()
{
    float min_distance[N] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 0; i < N - N % 8; i += 8)
            {
                // 取出原始积累的距离
                __m256 distance = _mm256_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[8] = {centroids[j][d]};
                __m256 centroid_d = _mm256_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m256 data_d = _mm256_load_ps(&data_align[d][i]);
                // 对每一数据该维度计算差值
                __m256 delta = _mm256_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm256_add_ps(distance, _mm256_mul_ps(delta, delta));
                // 存回
                _mm256_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 0; i < N; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}

void calculate_AVX_unaligned()
{
    float min_distance[N + 1] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 1; i < N - N % 8 + 1; i += 8)
            {
                // 取出原始积累的距离
                __m256 distance = _mm256_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[8] = {centroids[j][d]};
                __m256 centroid_d = _mm256_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m256 data_d = _mm256_loadu_ps(&data[d][i]);
                // 对每一数据该维度计算差值
                __m256 delta = _mm256_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm256_add_ps(distance, _mm256_mul_ps(delta, delta));
                // 存回
                _mm256_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 1; i < N + 1; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}

void calculate_AVX512_aligned()
{
    float min_distance[N] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 0; i < N - N % 16; i += 16)
            {
                // 取出原始积累的距离
                __m512 distance = _mm512_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[16] = {centroids[j][d]};
                __m512 centroid_d = _mm512_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m512 data_d = _mm512_load_ps(&data_align[d][i]);
                // 对每一数据该维度计算差值
                __m512 delta = _mm512_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm512_add_ps(distance, _mm512_mul_ps(delta, delta));
                // 存回
                _mm512_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 0; i < N; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}

void calculate_AVX512_unaligned()
{
    float min_distance[N + 1] = {0.0};
    for (int j = 0; j < K; j++)
    {
        // 各个点到各个聚类中心的距离
        float dis_k[N] = {0.0};
        for (int d = 0; d < D; d++)
        {
            for (int i = 1; i < N - N % 16 + 1; i += 16)
            {
                // 取出原始积累的距离
                __m512 distance = _mm512_loadu_ps(&dis_k[i]);
                // 构造质心的某一维度数据
                float tmp_centroid_d[16] = {centroids[j][d]};
                __m512 centroid_d = _mm512_loadu_ps(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                __m512 data_d = _mm512_loadu_ps(&data[d][i]);
                // 对每一数据该维度计算差值
                __m512 delta = _mm512_sub_ps(data_d, centroid_d);
                // 对每一数据该维度累加距离
                distance = _mm512_add_ps(distance, _mm512_mul_ps(delta, delta));
                // 存回
                _mm512_storeu_ps(&dis_k[i], distance);
            }
        }
        // 判断当前的每一个数据到该质心的距离是否是最小的
        for (int i = 1; i < N + 1; i++)
            if (dis_k[i] < min_distance[i])
                min_distance[i] = dis_k[i], cluster[i] = j;
    }
}
