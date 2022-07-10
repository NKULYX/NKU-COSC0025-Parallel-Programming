//
// Created by TTATT on 2022/5/5.
//

#include <iostream>
#include <vector>
#include "KMeansSerial.h"
#include "x86/KMeansSIMD.h"
#include "TestUtil.h"

using namespace std;

int main(){
    vector<KMeans*> kmeansList;
    KMeansSerial kMeansSerial = KMeansSerial(4);
    kmeansList.push_back(&kMeansSerial);
    KMeansSIMD kMeansSSE_UNALIGNED = KMeansSIMD(4, SIMD_UNALIGNED);
    kmeansList.push_back(&kMeansSSE_UNALIGNED);
    KMeansSIMD kMeansSSE_ALIGNED = KMeansSIMD(4, SIMD_ALIGNED);
    kmeansList.push_back(&kMeansSSE_ALIGNED);
    KMeansSIMD kMeansAVX_UNALIGNED = KMeansSIMD(4, SIMD_AVX_UNALIGNED);
    kmeansList.push_back(&kMeansAVX_UNALIGNED);
    KMeansSIMD kMeansAVX_ALIGNED = KMeansSIMD(4, SIMD_AVX_ALIGNED);
    kmeansList.push_back(&kMeansAVX_ALIGNED);
    KMeansSIMD kMeansAVX512_UNALIGNED = KMeansSIMD(4, SIMD_AVX512_UNALIGNED);
    kmeansList.push_back(&kMeansAVX512_UNALIGNED);
    KMeansSIMD kMeansAVX512_ALIGNED = KMeansSIMD(4, SIMD_AVX512_ALIGNED);
    kmeansList.push_back(&kMeansAVX512_ALIGNED);
    TestUtil::runSystemTest(kmeansList);
    return 0;
}


