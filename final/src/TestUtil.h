//
// Created by Lenovo on 2022/6/23.
//

#ifndef FINAL_TESTUTIL_H
#define FINAL_TESTUTIL_H


#include "KMeans.h"
#include <fstream>

class TestUtil {
    static float **getTestData(int n, int d, int k);
    static double getTestTime(KMeans& kmeans, float** testData, int n, int d);
    static void outputResult(int n, int d, int k, double time, std::fstream& result);
public:
    static void runTest(KMeans& kmeans);
};


#endif //FINAL_TESTUTIL_H
