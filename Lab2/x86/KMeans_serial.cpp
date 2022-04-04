#include <iostream>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include<windows.h>

using namespace std;

const int N = 4*4*4*4*4*4*4*4;
const int D = 4;
const int K = 4;
const int L = 100;
const int LOOP = 1;

float **data;  // 数据集
float data_align[D][N];
float centroids[K][D];  // 聚类中心
int cluster[N];  // 各数据所属类别
int cntCluster[K];  // 个聚类计数

void initData();
void initCentroids();
void calculate_serial();
void updateCentroids();

int main()
{
    long long int head, tail, freq;
    initData();
    initCentroids();
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int i=0; i < LOOP; i++)
    {
        calculate_serial();
        // updateCentroids();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq / LOOP << endl;
    system("pause");
}

void initData()
{
    data = new float*[D];
    for(int i=0; i < D; i++)
        data[i] = new float[N];
    for(int i=0; i<D; i++)
        for(int j=0; j<N; j++)
            data[i][j] = rand()*1.0/RAND_MAX * L , data_align[i][j] = rand()*1.0/RAND_MAX * L;
}

void initCentroids()
{
    for(int i=0; i<K; i++)
        for(int j=0; j<D; j++)
            centroids[i][j] = rand()*1.0/RAND_MAX * L;
}

void calculate_serial()
{
    for(int i = 0; i < N; i++)
    {
        float min_dis = L*L;
        for(int j = 0; j < K; j++)
        {
            float dis = 0;
            for(int d = 0; d < D; d++)
                dis += (data[d][i] - centroids[j][d]) * (data[d][i] - centroids[j][d]);
            if(dis < min_dis)
                min_dis = dis,cluster[i] = j,cntCluster[j]++;
        }
    }
}

void updateCentroids()
{
    for(int i=0; i< N; i++)
        for(int j=0; j<D; j++)
            centroids[cluster[i]][j] += data[j][i];
    for(int i=0;i<K;i++)
        for(int j=0;j<D;j++)
            centroids[i][j] /= cntCluster[i];
}