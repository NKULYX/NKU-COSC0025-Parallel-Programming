#include <iostream>
#include <sys/time.h>
using namespace std;

const int N = 10000;
const int D = 2;
const int K = 4;
const int L = 100;
const int LOOP = 100;

float data[D][N];  // 数据集
float centroids[K][D];  // 聚类中心
int cluster[N];  // 各数据所属类别
int cntCluster[K];  // 个聚类计数

void initData();
void initCentroids();
void calculate();
void updateCentroids();

int main()
{
    initData();
    initCentroids();
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int i=0; i < LOOP; i++)
    {
        calculate();
        updateCentroids();
    }
    gettimeofday(&end,NULL);
    cout<<"serial:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
    system("pause");
}

void initData()
{
    for(int i=0; i<D; i++)
        for(int j=0; j<N; j++)
            data[i][j] = rand()*1.0/RAND_MAX * L;
}

void initCentroids()
{
    for(int i=0; i<K; i++)
        for(int j=0; j<D; j++)
            centroids[i][j] = rand()*1.0/RAND_MAX * L;
}

void calculate()
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