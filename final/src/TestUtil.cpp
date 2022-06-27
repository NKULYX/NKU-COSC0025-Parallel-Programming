//
// Created by Lenovo on 2022/6/23.
//

#include "TestUtil.h"
#include <random>
#include <sys/time.h>
#include <iostream>

//#define _SYSTEM_TEST_
#define _UNIT_TEST_

using namespace std;

void TestUtil::runTest(KMeans& kmeans){
#ifdef _UNIT_TEST_
    int k = kmeans.getClusterNumber();
    int n = 10 * k;
    int d = 5;
    float** testData = getTestData(n,d,k);
    double time = getTestTime(kmeans, testData, n, d);
    cout << time << "ms" << endl;
    kmeans.printResult();
#else
    fstream result;
    result.open("result.csv");
    result<<"N"<<","<<"D"<<","<<"K"<<","<<"time"<<endl;
    for(int n = 100; n <= 1000; n++){
        for(int d = 2; d < 10; d++){
            float** testData = getTestData(n,d,kmeans.getClusterNumber());
            double time = getTestTime(kmeans, n ,d);
            outputResult(n, d, k, time, result);
        }
    }
    result.close();
#endif
}

/*
 * get the test data
 * @param n: the number of data
 * @param d: the dimension of data
 * @return: the test data
 */
float **TestUtil::getTestData(int n, int d, int k) {
    auto** tmpData = new float*[n];
    for(int i = 0; i < n; i++)
        tmpData[i] = new float[d];
    default_random_engine e;
    uniform_real_distribution<float> u1(0, 1000);
    uniform_real_distribution<float> u2(-50, 50);
    int step = n / k + 1;
    for(int i=0;i<k;i++){
        auto* tmpCenter = new float[d];
        for(int j=0;j<d;j++)
            tmpCenter[j] = u1(e);
        for(int j = i * step; j < (i + 1) * step && j < n; j++){
            for(int m=0; m < d; m++)
                tmpData[j][m] = tmpCenter[m] + u2(e);
        }
    }
    return tmpData;
}

double TestUtil::getTestTime(KMeans& kmeans, float** testData, int n, int d){
    int loop = n * d < 10000 ? 50 : 5;
    struct timeval start{};
    struct timeval end{};
    double time = 0;
    for(int i=0;i<loop;i++){
        kmeans.initData(testData, n, d);
        gettimeofday(&start, nullptr);
        kmeans.fit();
        gettimeofday(&end, nullptr);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    return time / loop;
}

void TestUtil::outputResult(int n, int d, int k, double time, std::fstream& result){
    result << n << d << k << time << endl;
}
