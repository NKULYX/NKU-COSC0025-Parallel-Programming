//
// Created by Lenovo on 2022/5/10.
//

#ifndef FINAL_KMEANSOMP_H
#define FINAL_KMEANSOMP_H

#include <string.h>
using namespace std;

class KMeansOMP {
    float **data;      // 数据
    int N;         // 数据量
    int D;         // 数据维度
    int K;         // 聚类数量
    int L;         // 迭代轮数
    float **centroids; // 质心
    int *clusterCount; // 每个数据所属的聚类
    int *clusterLabels;  // 聚类标签
    int method;    // 优化方法
    void initCentroidsRandom();
    void initCentroidsOptimize();
    void calculateSerial();
    void updateCentroids();
    float calculateDistance(float*, float*);
    void fitNormal();
    void fitKMeansPlusPlus();
    void fitSIMD();
    void fitPthread();
    void fitOMP();

public:
    KMeans(int K, int L, string method = "normal");
    ~KMeans();
    void initData(float**,int,int);
    float** getTestData(int,int);
    void fit();
    void printResult();
};


#endif //FINAL_KMEANSOMP_H
