//
// Created by Lenovo on 2022/7/2.
//

#include <iostream>
#include <vector>
#include "KMeansSerial.h"
#include "x86/KMeansPthread.h"
#include "TestUtil.h"

using namespace std;

int main(){
    vector<KMeans*> kmeansList;
    KMeansSerial kMeansSerial = KMeansSerial(4);
    kmeansList.push_back(&kMeansSerial);
    KMeansPthread kMeansSTATIC_DIV = KMeansPthread(4, PTHREAD_STATIC_DIV);
    kMeansSTATIC_DIV.setThreadNum(8);
    kmeansList.push_back(&kMeansSTATIC_DIV);
    KMeansPthread kMeansDYNAMIC_DIV = KMeansPthread(4, PTHREAD_DYNAMIC_DIV);
    kMeansDYNAMIC_DIV.setThreadNum(8);
    kmeansList.push_back(&kMeansDYNAMIC_DIV);
    KMeansPthread kMeansSTATIC_SIMD = KMeansPthread(4, PTHREAD_STATIC_SIMD);
    kMeansSTATIC_SIMD.setThreadNum(8);
    kmeansList.push_back(&kMeansSTATIC_SIMD);
    KMeansPthread kMeansDYNAMIC_SIMD = KMeansPthread(4, PTHREAD_DYNAMIC_SIMD);
    kMeansDYNAMIC_SIMD.setThreadNum(8);
    kmeansList.push_back(&kMeansDYNAMIC_SIMD);
    TestUtil::runSystemTest(kmeansList);
    return 0;
}