//
// Created by Lenovo on 2022/5/10.
//

#ifndef FINAL_KMEANSOMP_H
#define FINAL_KMEANSOMP_H

#include "KMeans.h"

#define OMP_STATIC 1
#define OMP_DYNAMIC 2
#define OMP_GUIDED 3
#define OMP_SIMD 4
#define OMP_STATIC_THREADS 5
#define OMP_DYNAMIC_THREADS 6

class KMeansOMP : public KMeans{
    int threadNum{};
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    void calculateSerial();
    void calculateStatic();
    void calculateDynamic();
    void calculateDynamicThreads();
    void calculateStaticThreads();
    void calculateOMPSIMD();
    void calculateGuided();
    float calculateDistanceSIMD(float *dataItem, float *centroidItem);
    void updateCentroidsSerial();
    void updateCentroidsStatic();
    void updateCentroidsDynamic();
    void updateCentroidsGuided();
    void updateCentroidsOMPSIMD();
    void updateCentroidsStaticThreads();
    void updateCentroidsDynamicThreads();

public:
    explicit KMeansOMP(int k, int method = 0);
    ~KMeansOMP();
    void fit() override;
    void setThreadNum(int threadNumber);

};


#endif //FINAL_KMEANSOMP_H
