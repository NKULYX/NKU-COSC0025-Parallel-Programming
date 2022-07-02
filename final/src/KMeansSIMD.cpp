//
// Created by Lenovo on 2022/6/28.
//


#include <xmmintrin.h> //SSE
#include <pmmintrin.h> //SSE3
#include <immintrin.h> //AVX、AVX2
#include <cstring>
#include "KMeansSIMD.h"

KMeansSIMD::KMeansSIMD(int k, int method) : KMeans(k, method) {
}

/*
 * change the memory of the data and centroids
 * in order to call destruct function of ~KMeans()
 */
KMeansSIMD::~KMeansSIMD(){
    data = new float*[this->N];
    for(int i = 0; i < this->N; i++)
        data[i] = new float[this->D];
    centroids = new float*[this->K];
    for(int i = 0; i < this->K; i++)
        centroids[i] = new float[this->D];
    clusterCount = new int[this->K];
}

/*
 * the function to execute cluster process
 * first initial the centroids
 * then iterate over the loop
 * calculate the nearest centroid of each point and change the cluster labels
 * last update the centroids
 */
void KMeansSIMD::fit() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

/*
 * change the memory management according to the method
 */
void KMeansSIMD::changeMemory() {
    if (method % 2 == 0) {
        // adjust the memory of data
        auto **newData = (float **) malloc(sizeof(float *) * this->N);
        for (int i = 0; i < this->N; i++) {
            newData[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        for (int i = 0; i < this->N; i++) {
            for (int j = 0; j < this->D; j++) {
                newData[i][j] = data[i][j];
            }
        }
        for (int i = 0; i < this->N; i++) {
            delete[] data[i];
        }
        delete[] data;
        data = newData;
        // adjust the memory of centroids
        centroids = (float **) malloc(sizeof(float *) * this->K);
        for(int i = 0; i < this->K; i++) {
            centroids[i] = (float *) _aligned_malloc(sizeof(float) * this->D, 64);
        }
        // adjust the memory of clusterCount
        clusterCount = (int *) _aligned_malloc(sizeof(int) * this->K, 64);
    }
}

/*
 * calculate the nearest centroid of each point
 * how to execute calculate() depends on the method
 */
void KMeansSIMD::calculate() {
    switch (method) {
        case SIMD_SSE_UNALIGNED:
        case SIMD_SSE_ALIGNED:
            calculateSSE();
            break;
        case SIMD_AVX_UNALIGNED:
        case SIMD_AVX_ALIGNED:
            calculateAVX();
            break;
        case SIMD_AVX512_UNALIGNED:
        case SIMD_AVX512_ALIGNED:
            calculateAVX512();
            break;
        default:
            break;
    }
}

/*
 * calculate the nearest centroid of each point using SSE
 */
void KMeansSIMD::calculateSSE() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistanceSSE(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

/*
 * calculate the distance between two points using SSE
 */
float KMeansSIMD::calculateDistanceSSE(float *dataItem, float *centroidItem) {
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 4; i+=4) {
        __m128 tmpData, centroid;
        if(this->method == SIMD_SSE_UNALIGNED){
            tmpData = _mm_loadu_ps(&dataItem[i]);
            centroid = _mm_loadu_ps(&centroidItem[i]);
        }
        else{
            tmpData = _mm_load_ps(&dataItem[i]);
            centroid = _mm_load_ps(&centroidItem[i]);
        }
        __m128 diff = _mm_sub_ps(tmpData, centroid);
        __m128 square = _mm_mul_ps(diff, diff);
        __m128 sum = _mm_hadd_ps(square, square);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        dis += _mm_cvtss_f32(sum);
    }
    for(int i = this->D - this->D % 4; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

/*
 * calculate the nearest centroid of each point using AVX
 */
void KMeansSIMD::calculateAVX() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistanceAVX(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

/*
 * calculate the distance between two points using AVX
 */
float KMeansSIMD::calculateDistanceAVX(float *dataItem, float *centroidItem) {
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 8; i+=8) {
        __m256 tmpData, centroid;
        if(this->method == SIMD_AVX_UNALIGNED){
            tmpData = _mm256_loadu_ps(&dataItem[i]);
            centroid = _mm256_loadu_ps(&centroidItem[i]);
        }
        else{
            tmpData = _mm256_load_ps(&dataItem[i]);
            centroid = _mm256_load_ps(&centroidItem[i]);
        }
        __m256 diff = _mm256_sub_ps(tmpData, centroid);
        __m256 square = _mm256_mul_ps(diff, diff);
        __m256 sum = _mm256_hadd_ps(square, square);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        dis += _mm256_cvtss_f32(sum);
    }
    for(int i = this->D - this->D % 8; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

/*
 * calculate the nearest centroid of each point using AVX512
 */
void KMeansSIMD::calculateAVX512() {
    for (int i = 0; i < this->N; i++) {
        float min = 1e9;
        int minIndex = 0;
        for (int k = 0; k < this->K; k++) {
            float dis = calculateDistanceAVX512(this->data[i], this->centroids[k]);
            if (dis < min) {
                min = dis;
                minIndex = k;
            }
        }
        this->clusterLabels[i] = minIndex;
    }
}

/*
 * calculate the distance between two points using AVX512
 */
float KMeansSIMD::calculateDistanceAVX512(float *dataItem, float *centroidItem) {
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 16; i+=16) {
        __m512 tmpData, centroid;
        if(this->method == SIMD_AVX512_UNALIGNED){
            tmpData = _mm512_loadu_ps(&dataItem[i]);
            centroid = _mm512_loadu_ps(&centroidItem[i]);
        }
        else{
            tmpData = _mm512_load_ps(&dataItem[i]);
            centroid = _mm512_load_ps(&centroidItem[i]);
        }
        __m512 diff = _mm512_sub_ps(tmpData, centroid);
        __m512 square = _mm512_mul_ps(diff, diff);
        __m128 sum1 = _mm512_extractf32x4_ps(square, 0);
        __m128 sum2 = _mm512_extractf32x4_ps(square, 1);
        __m128 sum3 = _mm512_extractf32x4_ps(square, 2);
        __m128 sum4 = _mm512_extractf32x4_ps(square, 3);
        sum1 = _mm_hadd_ps(sum1, sum3);
        sum2 = _mm_hadd_ps(sum2, sum4);
        sum1 = _mm_hadd_ps(sum1, sum2);
        sum1 = _mm_hadd_ps(sum1, sum1);
        sum1 = _mm_hadd_ps(sum1, sum1);
        dis += _mm_cvtss_f32(sum1);
    }
    for(int i = this->D - this->D % 16; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

/*
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansSIMD::updateCentroids() {
    switch (method) {
        case SIMD_SSE_UNALIGNED:
        case SIMD_SSE_ALIGNED:
            updateCentroidsSSE();
            break;
        case SIMD_AVX_UNALIGNED:
        case SIMD_AVX_ALIGNED:
            updateCentroidsAVX();
            break;
        case SIMD_AVX512_UNALIGNED:
        case SIMD_AVX512_ALIGNED:
            updateCentroidsAVX512();
            break;
        default:
            break;
    }
}

/*
 * update the centroids using SSE
 */
void KMeansSIMD::updateCentroidsSSE() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using SSE
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        if(method == SIMD_SSE_UNALIGNED){
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_loadu_ps(&this->data[i][j]);
                __m128 centroid = _mm_loadu_ps(&this->centroids[cluster][j]);
                __m128 sum = _mm_add_ps(tmpData, centroid);
                _mm_storeu_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_load_ps(&this->data[i][j]);
                __m128 centroid = _mm_load_ps(&this->centroids[cluster][j]);
                __m128 sum = _mm_add_ps(tmpData, centroid);
                _mm_store_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
    }
    // calculate the mean of the cluster using SSE
    for(int i=0;i<this->K;i++){
        if(method == SIMD_SSE_UNALIGNED){
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_loadu_ps(&this->centroids[i][j]);
                __m128 count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m128 mean = _mm_div_ps(tmpData, count);
                _mm_storeu_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 4;j+=4){
                __m128 tmpData = _mm_load_ps(&this->centroids[i][j]);
                __m128 count = _mm_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m128 mean = _mm_div_ps(tmpData, count);
                _mm_store_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 4;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
    }
}

/*
 * update the centroids using AVX
 */
void KMeansSIMD::updateCentroidsAVX() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using AVX
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        if(method == SIMD_AVX_UNALIGNED){
            for(int j=0;j<this->D - this->D % 8;j+=8){
                __m256 tmpData = _mm256_loadu_ps(&this->data[i][j]);
                __m256 centroid = _mm256_loadu_ps(&this->centroids[cluster][j]);
                __m256 sum = _mm256_add_ps(tmpData, centroid);
                _mm256_storeu_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 8;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 8;j+=8){
                __m256 tmpData = _mm256_load_ps(&this->data[i][j]);
                __m256 centroid = _mm256_load_ps(&this->centroids[cluster][j]);
                __m256 sum = _mm256_add_ps(tmpData, centroid);
                _mm256_store_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 8;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
    }
    // calculate the mean of the cluster using AVX
    for(int i=0;i<this->K;i++){
        if(method == SIMD_AVX_UNALIGNED){
            for(int j=0;j<this->D - this->D % 8;j+=8){
                __m256 tmpData = _mm256_loadu_ps(&this->centroids[i][j]);
                __m256 count = _mm256_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m256 mean = _mm256_div_ps(tmpData, count);
                _mm256_storeu_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 8;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 8;j+=8){
                __m256 tmpData = _mm256_load_ps(&this->centroids[i][j]);
                __m256 count = _mm256_load_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m256 mean = _mm256_div_ps(tmpData, count);
                _mm256_store_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 8;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
    }
}

/*
 * update the centroids using AVX512
 */
void KMeansSIMD::updateCentroidsAVX512() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using AVX512
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        if(method == SIMD_AVX512_UNALIGNED){
            for(int j=0;j<this->D - this->D % 16;j+=16){
                __m256 tmpData = _mm256_loadu_ps(&this->data[i][j]);
                __m256 centroid = _mm256_loadu_ps(&this->centroids[cluster][j]);
                __m256 sum = _mm256_add_ps(tmpData, centroid);
                _mm256_storeu_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 16;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 16;j+=16){
                __m256 tmpData = _mm256_load_ps(&this->data[i][j]);
                __m256 centroid = _mm256_load_ps(&this->centroids[cluster][j]);
                __m256 sum = _mm256_add_ps(tmpData, centroid);
                _mm256_store_ps(&this->centroids[cluster][j], sum);
            }
            for(int j=this->D - this->D % 16;j<this->D;j++){
                this->centroids[cluster][j] += this->data[i][j];
            }
        }
    }
    // calculate the mean of the cluster using AVX512
    for(int i=0;i<this->K;i++){
        if(method == SIMD_AVX512_UNALIGNED){
            for(int j=0;j<this->D - this->D % 16;j+=16){
                __m256 tmpData = _mm256_loadu_ps(&this->centroids[i][j]);
                __m256 count = _mm256_loadu_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m256 mean = _mm256_div_ps(tmpData, count);
                _mm256_storeu_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 16;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
        else{
            for(int j=0;j<this->D - this->D % 16;j+=16){
                __m256 tmpData = _mm256_load_ps(&this->centroids[i][j]);
                __m256 count = _mm256_load_ps(reinterpret_cast<const float *>(&this->clusterCount[i]));
                __m256 mean = _mm256_div_ps(tmpData, count);
                _mm256_store_ps(&this->centroids[i][j], mean);
            }
            for(int j=this->D - this->D % 16;j<this->D;j++){
                this->centroids[i][j] /= this->clusterCount[i];
            }
        }
    }
}
