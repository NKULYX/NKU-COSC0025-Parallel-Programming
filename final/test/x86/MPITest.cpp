//
// Created by Lenovo on 2022/7/5.
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "KMeansSerial.h"
#include "x86/KMeansMPI.h"
#include "TestUtil.h"

using namespace std;

int main(){
    MPI_Init(nullptr, nullptr);
    vector<KMeans*> kmeansList;
    KMeansSerial kMeansSerial = KMeansSerial(4);
    kmeansList.push_back(&kMeansSerial);
    KMeansMPI kMeansMPI_BLOCK = KMeansMPI(4,MPI_BLOCK);
    kmeansList.push_back(&kMeansMPI_BLOCK);
    KMeansMPI kMeansMPI_CYCLE = KMeansMPI(4,MPI_CYCLE);
    kmeansList.push_back(&kMeansMPI_CYCLE);
    KMeansMPI kMeansMPI_MASTER_SLAVE = KMeansMPI(4,MPI_MASTER_SLAVE);
    kmeansList.push_back(&kMeansMPI_MASTER_SLAVE);
    KMeansMPI kMeansMPI_SLAVE_SLAVE = KMeansMPI(4,MPI_SLAVE_SLAVE);
    kmeansList.push_back(&kMeansMPI_SLAVE_SLAVE);
    KMeansMPI kMeansMPI_PIPELINE = KMeansMPI(4,MPI_PIPELINE);
    kmeansList.push_back(&kMeansMPI_PIPELINE);
    KMeansMPI kMeansMPI_SIMD = KMeansMPI(4,MPI_SIMD);
    kmeansList.push_back(&kMeansMPI_SIMD);
    KMeansMPI kMeansMPI_OMP = KMeansMPI(4,MPI_OMP);
    kMeansMPI_OMP.setThreadNum(8);
    kmeansList.push_back(&kMeansMPI_OMP);
    KMeansMPI kMeansMPI_OMP_SIMD = KMeansMPI(4,MPI_OMP_SIMD);
    kMeansMPI_OMP_SIMD.setThreadNum(8);
    kmeansList.push_back(&kMeansMPI_OMP_SIMD);
    TestUtil::runSystemTest(kmeansList);
    MPI_Finalize();
    return 0;
}

