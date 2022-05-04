#include <iostream>
#include "KMeans.h"
using namespace std;

int main() {
    KMeans kmeans = KMeans(5);
    kmeans.fit();
    kmeans.printResult();
    return 0;
}
