//
// Created by Lenovo on 2022/6/23.
//

#include "TestUtil.h"
#include <random>
#include <sys/time.h>
#include <iostream>

using namespace std;

void TestUtil::runUnitTest(KMeans& kmeans) {
    int k = kmeans.getClusterNumber();
    int n = 1000;
    int d = 1000;
    float** testData = getTestData(n,d,k);
    double time = getTestTime(kmeans, testData, n, d);
    cout << time << "ms" << endl;
}

void TestUtil::runSystemTest(std::vector<KMeans*> kmeansList){
    ofstream result("./result.csv");
    result<<"N"<<","<<"D"<<","<<"K"<<","<<"time"<<endl;
    for(int n = 500, d = 500; n <= 4000 && d <= 4000; n+=500, d+=500 ){
        result << n << "," << d;
        cout << n << "," << d;
        float** testData = getTestData(n, d, (*kmeansList.begin())->getClusterNumber());
        for(auto& kmeans : kmeansList){
            double time = getTestTime(*kmeans, testData, n, d);
            result << "," << time;
            cout << ","  << time;
        }
        result << endl;
        cout << endl;
    }
    result.close();
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
//    int loop = n * d < 10000 ? 10 : 1;
    int loop = 1;
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
