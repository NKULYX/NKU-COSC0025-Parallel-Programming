//
// Created by Lenovo on 2022/7/2.
//

#include "KMeansPthread.h"
#include <pmmintrin.h> //SSE3
#include <cstring>

KMeansPthread::KMeansPthread(int k, int mehtod) : KMeans(k, mehtod) {
}

KMeansPthread::~KMeansPthread() = default;

/*
 * set the number of threads
 */
void KMeansPthread::setThreadNum(int threadNumber) {
    this->threadNum = threadNumber;
}

/*
 * change the memory management according to the method
 */
void KMeansPthread::changeMemory(){
}

/*
 * the function to execute cluster process
 * first initial the centroids
 * then iterate over the loop
 * calculate the nearest centroids of each point and change the cluster labels
 * last update the centroids
 */
void KMeansPthread::fit(){
    initCentroidsRandom();
    initThread();
}

/*
 * init the thread and semaphore
 */
void KMeansPthread::initThread() {
    void *(*threadFunc)(void *);
    switch(method){
        case PTHREAD_STATIC_DIV:
            threadFunc = threadFuncStaticDiv;
            break;
        case PTHREAD_DYNAMIC_DIV:
            threadFunc = threadFuncDynamicDiv;
            break;
        case PTHREAD_STATIC_SIMD:
            threadFunc = threadFuncStaticSIMD;
            break;
        case PTHREAD_DYNAMIC_SIMD:
            threadFunc = threadFuncDynamicSIMD;
            break;
        default:
            threadFunc = nullptr;
            break;
    }
    // init semaphore, barrier and lock
    sem_init(&sem, 0, 0);
    pthread_barrier_init(&barrier, nullptr, threadNum);
    pthread_mutex_init(&lock, nullptr);
    // create thread parameters
    auto* threadParams = new threadParam_t[threadNum];
    // create threads
    auto* threads = new pthread_t[threadNum];
    for(int i = 0; i < threadNum; i++){
        threadParams[i].thread_id = i;
        pthread_create(&threads[i], nullptr, threadFunc, (void *)(&threadParams[i]));
    }
    // execute the thread
    for(int i = 0; i < threadNum; i++){
        pthread_join(threads[i], nullptr);
    }
    // destroy the semaphore and barrier
    sem_destroy(&sem);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&lock);
    // destroy the thread parameters and threads
    delete[] threadParams;
    delete[] threads;
}

/*
 * each thread calculates the nearest centroid of some points depends on the thread_id
 * the thread 0 update the centroids
 */
void *KMeansPthread::threadFuncStaticDiv(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        for (int i = thread_id; i < this->N; i += threadNum) {
            float min = 1e9;
            int minIndex = 0;
            for(int k=0;k<this->K;k++) {
                float dis = calculateDistance(this->data[i], this->centroids[k]);
                if(dis < min) {
                    min = dis;
                    minIndex = k;
                }
            }
            this->clusterLabels[i] = minIndex;
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void *KMeansPthread::threadFuncDynamicDiv(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        while (taskIndex < this->N) {
            int begin = taskIndex;
            pthread_mutex_lock(&lock);
            taskIndex += taskNum;
            pthread_mutex_unlock(&lock);
            int end = taskIndex < this->N ? taskIndex : this->N;
            for (int i = begin; i < end; i++) {
                float min = 1e9;
                int minIndex = 0;
                for (int k = 0; k < this->K; k++) {
                    float dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void *KMeansPthread::threadFuncStaticSIMD(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        for (int i = thread_id; i < this->N; i += threadNum) {
            float min = 1e9;
            int minIndex = 0;
            for(int k=0;k<this->K;k++) {
                float dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
                if(dis < min) {
                    min = dis;
                    minIndex = k;
                }
            }
            this->clusterLabels[i] = minIndex;
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

void *KMeansPthread::threadFuncDynamicSIMD(void *param) {
    auto* threadParam = (threadParam_t*)param;
    int thread_id = threadParam->thread_id;
    for(int l = 0; l < this->L; l++){
        while (taskIndex < this->N) {
            int begin = taskIndex;
            pthread_mutex_lock(&lock);
            taskIndex += taskNum;
            pthread_mutex_unlock(&lock);
            int end = taskIndex < this->N ? taskIndex : this->N;
            for (int i = begin; i < end; i++) {
                float min = 1e9;
                int minIndex = 0;
                for (int k = 0; k < this->K; k++) {
                    float dis = calculateDistanceSIMD(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
        if(thread_id == 0){
            updateCentroids();
        }
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}

/*
 * calculate the nearest centroid of each point
 * how to execute calculate() depends on the method
 */
void KMeansPthread::calculate(){
}

float KMeansPthread::calculateDistanceSIMD(float* dataItem, float* centroidItem){
    float dis = 0;
    for(int i = 0; i < this->D - this->D % 4; i+=4) {
        __m128 tmpData, centroid;
        tmpData = _mm_loadu_ps(&dataItem[i]);
        centroid = _mm_loadu_ps(&centroidItem[i]);
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
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansPthread::updateCentroids(){
    switch(this->method){
        case PTHREAD_STATIC_DIV:
        case PTHREAD_DYNAMIC_DIV:
            updateCentroidsSerial();
            break;
        case PTHREAD_STATIC_SIMD:
        case PTHREAD_DYNAMIC_SIMD:
            updateCentroidsSIMD();
            break;
    }
}

void KMeansPthread::updateCentroidsSerial() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        for(int j=0;j<this->D;j++){
            this->centroids[cluster][j] += this->data[i][j];
        }
        this->clusterCount[cluster]++;
    }
    // calculate the mean of the cluster
    for(int i=0;i<this->K;i++){
        for(int j=0;j<this->D;j++){
            this->centroids[i][j] /= (float)this->clusterCount[i];
        }
    }
}

void KMeansPthread::updateCentroidsSIMD() {
// initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using SSE
    for(int i=0;i<this->N;i++){
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
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
    // calculate the mean of the cluster using SSE
    for(int i=0;i<this->K;i++){
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
}


