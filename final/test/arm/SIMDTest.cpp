//
// Created by TTATT on 2022/5/5.
//

#include <iostream>
#include <vector>
#include "KMeansSerial.h"
#include "arm/KMeansSIMD.h"
#include "TestUtil.h"

using namespace std;

int main(){
    vector<KMeans*> kmeansList;
    KMeansSerial kMeansSerial = KMeansSerial(4);
    kmeansList.push_back(&kMeansSerial);
    KMeansSIMD kMeansSSE_UNALIGNED = KMeansSIMD(4, SIMD_UNALIGNED);
    kmeansList.push_back(&kMeansSSE_UNALIGNED);
    KMeansSIMD kMeansSSE_ALIGNED = KMeansSIMD(4, SIMD_ALIGNED);
    TestUtil::runSystemTest(kmeansList);
    return 0;
}


