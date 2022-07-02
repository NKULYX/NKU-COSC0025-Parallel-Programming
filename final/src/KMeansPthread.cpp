//
// Created by Lenovo on 2022/7/2.
//

#include <functional>
#include "KMeansPthread.h"

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
    return nullptr;
}

void *KMeansPthread::threadFuncDynamicDiv(void *param) {
    return nullptr;
}

void *KMeansPthread::threadFuncStaticSIMD(void *param) {
    return nullptr;
}

void *KMeansPthread::threadFuncDynamicSIMD(void *param) {
    return nullptr;
}

/*
 * calculate the nearest centroid of each point
 * how to execute calculate() depends on the method
 */
void KMeansPthread::calculate(){
    switch (method) {

    }
}

/*
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansPthread::updateCentroids(){
    switch (method) {

    }
}

