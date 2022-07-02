#ifndef FINAL_SERIALKMEANS_H
#define FINAL_SERIALKMEANS_H

#include "KMeans.h"

class KMeansSerial:public KMeans{
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;

public:
    explicit KMeansSerial(int k, int method = 0);
    ~KMeansSerial();
    void fit() override;


};







#endif // FINAL_SERIALKMEANS_H