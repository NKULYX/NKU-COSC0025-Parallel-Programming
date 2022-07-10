//
// Created by Lenovo on 2022/6/28.
//

#ifndef FINAL_KMEANSSIMD_H
#define FINAL_KMEANSSIMD_H

#include "KMeans.h"

#define SIMD_UNALIGNED 1
#define SIMD_ALIGNED 2

class KMeansSIMD : public KMeans{
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void calculateSIMD();
    float calculateDistanceSIMD(float *dataItem, float *centroidItem);
    void updateCentroidsSIMD();

public:
    explicit KMeansSIMD(int k, int method = 0);
    ~KMeansSIMD();
    void fit() override;

};


#endif //FINAL_KMEANSSIMD_H
