//
// Created by TTATT on 2022/5/5.
//

#include <iostream>
#include "KMeans.h"
using namespace std;

int main() {
    KMeans kmeans = KMeans(5, 100);
    float** testData = kmeans.getTestData(1000,4);
    kmeans.initData(testData,1000,4);
    kmeans.fit();
    kmeans.printResult();
    return 0;
}
