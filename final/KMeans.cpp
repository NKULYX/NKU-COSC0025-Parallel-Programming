//
// Created by TTATT on 2022/5/5.
//

#include "KMeans.h"

#include <iostream>
#include <cstring>
#include <random>


KMeans::KMeans(int K) {
    KMeans(K, "normal");
}

KMeans::KMeans(int K, string method) {
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

}

void KMeans::fitKMeansPlusPlus() {

}

void KMeans::fitSIMD() {

}

void KMeans::fitPthread() {

}

void KMeans::fitOMP() {

}
