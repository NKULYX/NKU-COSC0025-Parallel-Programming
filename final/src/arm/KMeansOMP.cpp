//
// Created by Lenovo on 2022/5/10.
//


#include <arm_neon.h>
#include <omp.h>
#include <cstring>
#include "KMeansOMP.h"

KMeansOMP::KMeansOMP(int k, int method) : KMeans(k, method) {
}

KMeansOMP::~KMeansOMP()= default;

void KMeansOMP::changeMemory() {}

void KMeansOMP::setThreadNum(int threadNumber){
    this->threadNum = threadNumber;
}

/*
 * the function to execute cluster process
 * first initial the centroids
 * then iterate over the loop
 * calculate the nearest centroid of each point and change the cluster labels
 * last update the centroids
 */
void KMeansOMP::fit() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

void KMeansOMP::calculate() {
    switch(method){
        case OMP_STATIC:
            calculateStatic();
            break;
        case OMP_DYNAMIC:
            calculateDynamic();
            break;
        case OMP_GUIDED:
            calculateGuided();
            break;
        case OMP_SIMD:
            calculateOMPSIMD();
            break;
        case OMP_STATIC_THREADS:
            calculateStaticThreads();
            break;
        case OMP_DYNAMIC_THREADS:
            calculateDynamicThreads();
            break;
        default:
            calculateSerial();
            break;
    }
}

void KMeansOMP::calculateSerial(){
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

void KMeansOMP::calculateStatic() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansOMP::calculateDynamic() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(dynamic)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansOMP::calculateGuided() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansOMP::calculateOMPSIMD() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

float KMeansOMP::calculateDistanceSIMD(float *dataItem, float *centroidItem) {
    float dis = 0;
    for (int i = 0; i < this->D - this->D % 4; i += 4) {
        float32x4_t tmpData, centroid;
        tmpData = vmovq_n_f32(&dataItem[i]);
        centroid = vmovq_n_f32(&centroidItem[i]);
        float32x4_t diff = vsubq_f32(tmpData, centroid);
        float32x4_t square = vmulq_f32(diff, diff);
        float sum[4];
        vst1q_f32(sum, square);
        dis += sum[0] + sum[1] + sum[2] + sum[3];
    }
    for (int i = this->D - this->D % 4; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

void KMeansOMP::calculateStaticThreads() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansOMP::calculateDynamicThreads() {
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            for (i = 0; i < this->N; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

/*
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansOMP::updateCentroids() {
    switch(method){
        case OMP_STATIC:
            updateCentroidsStatic();
            break;
        case OMP_DYNAMIC:
            updateCentroidsDynamic();
            break;
        case OMP_GUIDED:
            updateCentroidsGuided();
            break;
        case OMP_SIMD:
            updateCentroidsOMPSIMD();
            break;
        case OMP_STATIC_THREADS:
            updateCentroidsStaticThreads();
            break;
        case OMP_DYNAMIC_THREADS:
            updateCentroidsDynamicThreads();
            break;
        default:
            updateCentroidsSerial();
            break;
    }
}

void KMeansOMP::updateCentroidsSerial() {
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
            this->centroids[i][j] /= (float)this->clusterCount[i];
        }
    }
}

void KMeansOMP::updateCentroidsStatic() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(static)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsDynamic() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(dynamic)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(dynamic)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsGuided() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(guided)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsOMPSIMD() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
    float32x4_t tmpData, centroid, sum;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster, tmpData, centroid, sum) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
                this->clusterCount[cluster]++;
                for (j = 0; j < this->D - this->D % 4; j += 4) {
                    tmpData = vmovq_n_f32(&this->data[i][j]);
                    centroid = vmovq_n_f32(&this->centroids[cluster][j]);
                    sum = vaddq_f32(tmpData, centroid);
                    vst1q_f32(&this->centroids[cluster][j], sum);
                }
                for (j = this->D - this->D % 4; j < this->D; j++) {
                    this->centroids[cluster][j] += this->data[i][j];
                }
            }
        }
#pragma omp for schedule(guided)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsStaticThreads() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(guided)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansOMP::updateCentroidsDynamicThreads() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i,j,cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(guided)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

