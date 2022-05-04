#include "KMeans.h"

int main() {
    KMeans<int> kmeans = KMeans<int>(5);
    kmeans.print();
    return 0;
}
