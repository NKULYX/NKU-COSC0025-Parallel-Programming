//
// Created by TTATT on 2022/5/5.
//

#include "KMeansSerial.h"

#include <cstring>

KMeansSerial::KMeansSerial(int k, int method) : KMeans(k, method) {
}

KMeansSerial::~KMeansSerial() = default;

/*
 * the function to execute cluster process
 * first initial the centroids
 * then iterate over the loop
 * calculate the nearest centroid of each point and change the cluster labels
 * last update the centroids
 */
void KMeansSerial::fit() {
    initCentroidsRandom();
    for(int i=0; i<this->L;i++){
        calculate();
        updateCentroids();
    }
}

/*
 * calculate the nearest centroid of each point serially
 */
void KMeansSerial::calculate() {
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
void KMeansSerial::updateCentroids() {
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

void KMeansSerial::changeMemory() {

}
