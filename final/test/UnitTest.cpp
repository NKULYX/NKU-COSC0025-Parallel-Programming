//
// Created by TTATT on 2022/5/5.
//

#include "KMeansSerial.h"
#include "KMeansSIMD.h"
#include "TestUtil.h"

using namespace std;

int main() {
//    KMeansSerial kMeans = KMeansSerial(4);
    KMeansSIMD kMeans = KMeansSIMD(4,AVX_ALIGNED);
    TestUtil::runUnitTest(kMeans);
    return 0;
}
