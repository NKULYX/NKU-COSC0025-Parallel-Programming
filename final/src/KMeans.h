/*
 * Created by TTATT on 2022/6/17.
 * This class is the base class for KMeans
 * It provides the basic member variables and member functions
 */

#ifndef FINAL_KMEANS_H
#define FINAL_KMEANS_H

class KMeans {

    virtual void calculate() = 0;

    virtual void updateCentroids() = 0;


protected:
    float **data{};                     // the description of data
    int N{};                            // the number of the data
    int D{};                            // the dimension of the data
    int K{};                              // the number of centroids
    int L = 500;                        // the loop for iteration
    float **centroids{};                // the description of centroids
    int *clusterCount{};                  // the number of data in each cluster
    int *clusterLabels{};               // the label of cluster
    int method;                         // optimize method

    void initCentroidsRandom();

    float calculateDistance(const float *, const float *) const;

public:
    KMeans(int k);

    KMeans(int k, int method);

    ~KMeans();

    void initData(float ** data, int n, int d);

    virtual void fit() {};

    void printResult();

    int getClusterNumber();
};


#endif //FINAL_KMEANS_H
