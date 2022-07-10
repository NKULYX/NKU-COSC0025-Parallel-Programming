//
// Created by Lenovo on 2022/6/28.
//


#include <arm_neon.h>
#include <cstring>
#include <cassert>
#include "KMeansSIMD.h"

KMeansSIMD::KMeansSIMD(int k, int method) : KMeans(k, method) {
}

/*
 * change the memory of the data and centroids
 * in order to call destruct function of ~KMeans()
 */
KMeansSIMD::~KMeansSIMD(){
    data = new float*[this->N];
    for(int i = 0; i < this->N; i++)
        data[i] = new float[this->D];
    centroids = new float*[this->K];
    for(int i = 0; i < this->K; i++)
        centroids[i] = new float[this->D];
    clusterCount = new int[this->K];
}

/*
 * the function to execute cluster process
 * first initial the centroids
 * then iterate over the loop
 * calculate the nearest centroid of each point and change the cluster labels
 * last update the centroids
 */
void KMeansSIMD::fit() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

/*
 * change the memory management according to the method
 */
void KMeansSIMD::changeMemory() {
    if (method % 2 == 0) {
        // adjust the memory of data
        auto **newData = (float **) malloc(sizeof(float *) * this->N);
        for (int i = 0; i < this->N; i++) {
            newData[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        for (int i = 0; i < this->N; i++) {
            for (int j = 0; j < this->D; j++) {
                newData[i][j] = data[i][j];
            }
        }
        for (int i = 0; i < this->N; i++) {
            delete[] data[i];
        }
        delete[] data;
        data = newData;
        // adjust the memory of centroids
        centroids = (float **) malloc(sizeof(float *) * this->K);
        for(int i = 0; i < this->K; i++) {
            centroids[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        // adjust the memory of clusterCount
        clusterCount = (int *) _aligned_malloc(sizeof(int) * this->K, 64);
    }
}

/*
 * calculate the nearest centroid of each point
 * how to execute calculate() depends on the method
 */
void KMeansSIMD::calculate() {
    switch (method) {
        case SIMD_UNALIGNED:
        case SIMD_ALIGNED:
            calculateSIMD();
            break;
        default:
            break;
    }
}

/*
 * calculate the nearest centroid of each point using SSE
 */
void KMeansSIMD::calculateSIMD() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

/*
 * calculate the distance between two points using SSE
 */
float KMeansSIMD::calculateDistanceSIMD(float *dataItem, float *centroidItem) {
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

/*
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansSIMD::updateCentroids() {
    switch (method) {
        case SIMD_UNALIGNED:
        case SIMD_ALIGNED:
            updateCentroidsSIMD();
            break;
        default:
            break;
    }
}

/*
 * update the centroids using SSE
 */
void KMeansSIMD::updateCentroidsSIMD() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using SSE
    for (int i = 0; i < this->N; i++) {
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            float32x4_t tmpData = vmovq_n_f32(&this->data[i][j]);
            float32x4_t centroid = vmovq_n_f32(&this->centroids[cluster][j]);
            float32x4_t sum = vaddq_f32(tmpData, centroid);
            vst1q_f32(&this->centroids[cluster][j], sum);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[cluster][j] += this->data[i][j];
        }
    }
    // calculate the mean of the cluster using SSE
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            float32x4_t tmpData = vmovq_n_f32(&this->centroids[i][j]);
            float32x4_t count = vmovq_n_f32(reinterpret_cast<const float *>(&this->clusterCount[i]));
            float32x4_t mean = vdivq_f32(tmpData, count);
            vst1q_f32(&this->centroids[cluster][j], sum);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[i][j] /= (float) this->clusterCount[i];
        }
    }
}
