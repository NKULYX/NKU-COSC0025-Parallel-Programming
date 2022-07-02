//
// Created by Lenovo on 2022/6/28.
//

#ifndef FINAL_KMEANSSIMD_H
#define FINAL_KMEANSSIMD_H

#include "KMeans.h"

#define SSE_UNALIGNED 1
#define SSE_ALIGNED 2
#define AVX_UNALIGNED 3
#define AVX_ALIGNED 4
#define AVX512_UNALIGNED 5
#define AVX512_ALIGNED 6

class KMeansSIMD : public KMeans{
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void calculateSSE();
    void calculateAVX();
    void calculateAVX512();
    float calculateDistanceSSE(float *dataItem, float *centroidItem);
    float calculateDistanceAVX(float *dataItem, float *centroidItem);
    float calculateDistanceAVX512(float *dataItem, float *centroidItem);
    void updateCentroidsSSE();
    void updateCentroidsAVX();
    void updateCentroidsAVX512();

public:
    explicit KMeansSIMD(int k, int method = 0);
    ~KMeansSIMD();
    void fit() override;

};


#endif //FINAL_KMEANSSIMD_H
