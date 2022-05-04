#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <string>
using namespace std;

template <typename T>
class KMeans
{
    T **data;      // 数据
    int N;         // 数据量
    int D;         // 数据维度
    int K;         // 聚类数量
    int L;         // 迭代轮数
    T **centroids; // 质心
    int *cluster;  // 聚类标签
    int method;    // 优化方法
public:
    explicit KMeans(int K);
    KMeans(int K, string method);
    ~KMeans();
    void initData();
    void initCentroids();
    void run();
    void print();
};

template <typename T>
KMeans<T>::KMeans(int K)
{
    KMeans(K, "normal");
}

template <typename T>
KMeans<T>::KMeans(int K, string method)
{
    this->K = K;
    if(method == "normal")
        this->method = 0;
    else if(method == "KMeans++")
        this->method = 1;
    else if(method == "SIMD")
        this->method = 2;
    else if(method == "pthread")
        this->method = 3;
    else if(method == "omp")
        this->method = 4;
    else
        this->method = 0;
}

template <typename T>
KMeans<T>::~KMeans()
{
    //dtor
}

template <typename T>
void KMeans<T>::print()
{
    cout<<"Hello"<<endl;
}




#endif // KMEANS_H