//
// Created by Lenovo on 2022/6/28.
//

#ifndef FINAL_KMEANSSIMD_H
#define FINAL_KMEANSSIMD_H

#include "KMeans.h"

#define SIMD_UNALIGNED 1
#define SIMD_ALIGNED 2
#define SIMD_AVX_UNALIGNED 3
#define SIMD_AVX_ALIGNED 4
#define SIMD_AVX512_UNALIGNED 5
#define SIMD_AVX512_ALIGNED 6

class KMeansSIMD : public KMeans{
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void calculateSIMD();
    void calculateAVX();
    void calculateAVX512();
    float calculateDistanceSIMD(float *dataItem, float *centroidItem);
    float calculateDistanceAVX(float *dataItem, float *centroidItem);
    float calculateDistanceAVX512(float *dataItem, float *centroidItem);
    void updateCentroidsSIMD();
    void updateCentroidsAVX();
    void updateCentroidsAVX512();

public:
    explicit KMeansSIMD(int k, int method = 0);
    ~KMeansSIMD();
    void fit() override;

};


#endif //FINAL_KMEANSSIMD_H
