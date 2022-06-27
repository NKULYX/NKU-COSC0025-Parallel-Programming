#ifndef FINAL_SERIALKMEANS_H
#define FINAL_SERIALKMEANS_H

#include <string>
#include "KMeans.h"

using namespace std;

class KMeansSerial:public KMeans{
    void calculate() override;
    void updateCentroids() override;

public:
    explicit KMeansSerial(int K, int method = 0);
    ~KMeansSerial();
    void fit() override;


};







#endif // FINAL_SERIALKMEANS_H