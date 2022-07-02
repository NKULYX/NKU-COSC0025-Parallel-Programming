//
// Created by Lenovo on 2022/7/2.
//

#ifndef FINAL_KMEANSPTHREAD_H
#define FINAL_KMEANSPTHREAD_H

#include "KMeans.h"
#include <pthread.h>
#include <semaphore.h>

#define PTHREAD_STATIC_DIV 1
#define PTHREAD_DYNAMIC_DIV 2
#define PTHREAD_STATIC_SIMD 3
#define PTHREAD_DYNAMIC_SIMD 4


class KMeansPthread : public KMeans{

    typedef struct
    {
        int thread_id;
    } threadParam_t;

    int threadNum{};
    sem_t sem{};
    pthread_barrier_t barrier{};
    pthread_mutex_t lock{};
    void calculate() override;
    void updateCentroids() override;
    void changeMemory() override;
    static void* threadFuncStaticDiv(void* param);
    static void* threadFuncDynamicDiv(void* param);
    static void* threadFuncStaticSIMD(void* param);
    static void* threadFuncDynamicSIMD(void* param);

public:
    explicit KMeansPthread(int k, int mehtod = 0);
    ~KMeansPthread();
    void fit() override;
    void setThreadNum(int threadNumber);

    void initThread();
};


#endif //FINAL_KMEANSPTHREAD_H
