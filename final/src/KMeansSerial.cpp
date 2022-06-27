//
// Created by TTATT on 2022/5/5.
//

#include "KMeansSerial.h"

#include <cstring>

KMeansSerial::KMeansSerial(int K, int method) : KMeans(K, method) {
}

KMeansSerial::~KMeansSerial() = default;

void KMeansSerial::fit() {
    initCentroidsRandom();
    for(int i=0; i<this->L;i++){
        calculate();
        updateCentroids();
    }
}

/*
 * calculate serially
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
