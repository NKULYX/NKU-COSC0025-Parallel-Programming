//
// Created by TTATT on 2022/5/5.
//

#include "KMeans.h"

#include <iostream>
#include <cstring>
#include <random>

KMeans::KMeans(int K, int L, string method) {
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

    this->clusterCount = new int[K];
    memset(this->clusterCount, 0, sizeof(int) * K);
}

KMeans::~KMeans() {
    for(int i = 0; i < this->N; i++)
        delete[] data[i];
    delete[] data;
    for(int i = 0; i < this->K; i++)
        delete[] centroids[i];
    delete[] centroids;
    delete[] clusterLabels;
    delete[] clusterCount;
}

void KMeans::initData(float** inputData, int n, int d) {
    this->N = n;
    this->D = d;
    this->data = new float*[this->N];
    for(int i = 0; i < this->N; i++)
    {
        this->data[i] = new float[this->D];
        for(int j = 0; j < this->D; j++)
            this->data[i][j] = inputData[i][j];
    }
    this->clusterLabels = new int[this->N];
}

void KMeans::initCentroidsRandom() {
    this->centroids = new float*[this->K];
    default_random_engine e;
    uniform_int_distribution<unsigned> u(0, this->N - 1);
    for(int i = 0; i < this->K; i++)
    {
        this->centroids[i] = new float[this->D];
        int index = u(e) % this->N;
        for(int j = 0; j < this->D; j++)
            this->centroids[i][j] = this->data[index][j];
    }
}

void KMeans::initCentroidsOptimize() {

}

void KMeans::fit() {
    switch(this->method)
    {
        case 0:
            this->fitNormal();
            break;
        case 1:
            this->fitKMeansPlusPlus();
            break;
        case 2:
            this->fitSIMD();
            break;
        case 3:
            this->fitPthread();
            break;
        case 4:
            this->fitOMP();
            break;
        default:
            this->fitNormal();
            break;
    }
}

void KMeans::printResult() {
    for(int i = 0; i < this->K; i++)
        cout << "Cluster " << i << ": " << this->clusterCount[i] << " points" << endl;
}

void KMeans::fitNormal() {
    initCentroidsRandom();
    for(int i=0; i<this->L;i++){
        calculateSerial();
        updateCentroids();
    }
}

void KMeans::fitKMeansPlusPlus() {

}

void KMeans::fitSIMD() {

}

void KMeans::fitPthread() {

}

void KMeans::fitOMP() {

}

/*
 * calculate serially
 */
void KMeans::calculateSerial() {
    for(int i=0;i<this->N;i++){
        float min = 1e9;
        int minIndex = 0;
        for(int k=0;k<this->K;k++) {
            float dis = calculateDistance(this->data[i], this->centroids[k]);
            if(dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

/*
 * update the centroids serially
 */
void KMeans::updateCentroids() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        for(int j=0;j<this->D;j++){
            this->centroids[cluster][j] += this->data[i][j];
        }
        this->clusterCount[cluster]++;
    }
    // calculate the mean of the cluster
    for(int i=0;i<this->K;i++){
        for(int j=0;j<this->D;j++){
            this->centroids[i][j] /= this->clusterCount[i];
        }
    }
}

/*
 * calculate the distance between a data and a centroid
 * @param data: the data
 * @param centroid: the centroid
 * @return: the distance
 */
float KMeans::calculateDistance(float *data, float *centroid) {
    float dis = 0;
    for(int i = 0; i < this->D; i++)
        dis += (data[i] - centroid[i]) * (data[i] - centroid[i]);
    return dis;
}

/*
 * get the test data
 * @param n: the number of data
 * @param d: the dimension of data
 * @return: the test data
 */
float **KMeans::getTestData(int n, int d) {
    auto** tmpData = new float*[n];
    for(int i = 0; i < n; i++)
        tmpData[i] = new float[d];
    default_random_engine e;
    uniform_real_distribution<float> u1(0, 1000);
    uniform_real_distribution<float> u2(-50, 50);
    int step = n / this->K + 1;
    for(int i=0;i<this->K;i++){
        auto* tmpCenter = new float[d];
        for(int j=0;j<d;j++)
            tmpCenter[j] = u1(e);
        for(int j = i * step; j < (i + 1) * step && j < n; j++){
            for(int m=0; m < d; m++)
                tmpData[j][m] = tmpCenter[m] + u2(e);
        }
    }
    return tmpData;
}
