//
// Created by Lenovo on 2022/7/2.
//

#include <iostream>
#include <vector>
#include "KMeansSerial.h"
#include "KMeansOMP.h"
#include "TestUtil.h"

using namespace std;

int main(){
    vector<KMeans*> kmeansList;
    KMeansSerial kMeansSerial = KMeansSerial(4);
    kmeansList.push_back(&kMeansSerial);
    KMeansOMP kMeansOMP_STATIC = KMeansOMP(4,OMP_STATIC);
    kMeansOMP_STATIC.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_STATIC);
    KMeansOMP kMeansOMP_DYNAMIC = KMeansOMP(4,OMP_DYNAMIC);
    kMeansOMP_DYNAMIC.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_DYNAMIC);
    KMeansOMP kMeansOMP_GUIDED = KMeansOMP(4,OMP_GUIDED);
    kMeansOMP_GUIDED.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_GUIDED);
    KMeansOMP kMeansOMP_SIMD = KMeansOMP(4,OMP_SIMD);
    kMeansOMP_SIMD.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_SIMD);
    KMeansOMP kMeansOMP_STATIC_THREADS = KMeansOMP(4,OMP_STATIC_THREADS);
    kMeansOMP_STATIC_THREADS.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_STATIC_THREADS);
    KMeansOMP kMeansOMP_DYNAMIC_THREADS = KMeansOMP(4,OMP_DYNAMIC_THREADS);
    kMeansOMP_DYNAMIC_THREADS.setThreadNum(8);
    kmeansList.push_back(&kMeansOMP_DYNAMIC_THREADS);
    TestUtil::runSystemTest(kmeansList);
    return 0;
}