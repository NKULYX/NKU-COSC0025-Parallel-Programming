//
// Created by TTATT on 2022/5/5.
//

#include <iostream>
#include "KMeansSerial.h"
#include "TestUtil.h"
using namespace std;

int main() {
    KMeansSerial kMeans = KMeansSerial(4);
    TestUtil::runTest(kMeans);
    return 0;
}
