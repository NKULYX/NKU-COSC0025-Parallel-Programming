//
// Created by Lenovo on 2022/6/23.
//

#ifndef FINAL_TESTUTIL_H
#define FINAL_TESTUTIL_H


#include "KMeans.h"
#include <fstream>
#include <vector>

class TestUtil {
    static float **getTestData(int n, int d, int k);
    static double getTestTime(KMeans& kmeans, float** testData, int n, int d);
public:
    static void runSystemTest(std::vector<KMeans*> kmeansList);
    static void runUnitTest(KMeans& kmeans);
};


#endif //FINAL_TESTUTIL_H
