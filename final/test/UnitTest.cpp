//
// Created by TTATT on 2022/5/5.
//

#include "KMeansSerial.h"
#include "KMeansSIMD.h"
#include "KMeansPthread.h"
#include "KMeansOMP.h"
#include "TestUtil.h"

using namespace std;

int main() {
//    KMeansSerial kMeans = KMeansSerial(4);
//    KMeansSIMD kMeans = KMeansSIMD(4,SIMD_AVX_ALIGNED);
//    KMeansPthread kMeans = KMeansPthread(4);
    KMeansOMP kMeans = KMeansOMP(4);
    TestUtil::runUnitTest(kMeans);
    return 0;
}
