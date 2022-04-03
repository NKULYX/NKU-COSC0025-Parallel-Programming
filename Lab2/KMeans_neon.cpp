#include <iostream>
#include <sys/time.h>
#include <arm_neon.h>

const int N = 10000;
const int D = 2;
const int K = 4;
const int L = 100;
const int LOOP = 100;

float data[D][N];  // 数据集  转置
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
    cout<<"parallel:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
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
    for(int i = 0; i < N - N % 4; i+=4)
    {
        float tmp[4] = {L*L, L*L, L*L, L*L}; 
        float32x4_t min_dis = vld1q_f32(tmp);
        for(int j = 0; j < K; j++)
        {
            float32x4_t distance = vdupq_n_f32(0.0);
            for(int d = 0; d < D; d++)
            {
                // 构造质心的某一维度数据
                float tmp_centroid_d[4] = {centroids[j][d], centroids[j][d], centroids[j][d], centroids[j][d]};
                float32x4_t centroid_d = vld1q_f32(tmp_centroid_d);
                // 一次取出四个元素的某一维度数据
                float32x4_t data_d = vld1q_f32(&data[d][i]);
                // 对每一数据该维度计算差值
                float32x4_t delta = vsubq_f32(data_d,centroid_d);
                // 对每一数据该维度累加距离
                distance = vmlaq_f32(distance, delta, delta);
            }
            // 判断当前的每一个数据到该质心的距离是否是最小的
            float disK[4];
            vst1q_f32(disK, distance);
            for(int k = 0; k < 4; k++)
            {
                if(disK[k] < min_dis[k])
                {
                    min_dis[k] = disK[k];
                    cluster[i+k] = j;
                }
            }
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
