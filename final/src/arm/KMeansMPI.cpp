//
// Created by Lenovo on 2022/7/5.
//


#include <mpi.h>
#include <cmath>
#include <arm_neon.h>
#include <cstring>
#include "KMeansMPI.h"

KMeansMPI::KMeansMPI(int k, int method) : KMeans(k, method) {
}

KMeansMPI::~KMeansMPI() {
    delete[] dataMemory;
    delete[] centroidMemory;
    data = new float *[this->N];
    for (int i = 0; i < this->N; i++) {
        data[i] = new float[this->D];
    }
    centroids = new float *[this->K];
    for (int i = 0; i < this->K; i++) {
        centroids[i] = new float[this->D];
    }
}

void KMeansMPI::setThreadNum(int threadNumber) {
    this->threadNum = threadNumber;
}

/*
 * change the memory management
 */
void KMeansMPI::changeMemory() {
    dataMemory = new float[this->N * this->D];
    for (int i = 0; i < this->N; i++) {
        for (int j = 0; j < this->D; j++) {
            dataMemory[i * this->D + j] = this->data[i][j];
        }
    }
    for (int i = 0; i < this->N; i++) {
        delete[] data[i];
    }
    delete[] data;
    for (int i = 0; i < this->N; i++) {
        data[i] = &dataMemory[i * this->D];
    }
    centroidMemory = new float[this->K * this->D];
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D; j++) {
            centroidMemory[i * this->D + j] = this->centroids[i][j];
        }
    }
    for (int i = 0; i < this->K; i++) {
        delete[] centroids[i];
    }
    delete[] centroids;
    for (int i = 0; i < this->K; i++) {
        centroids[i] = &centroidMemory[i * this->D];
    }
}

/*
 * fit according to the method
 */
void KMeansMPI::fit() {
    switch (method) {
        case MPI_BLOCK:
            fitMPI_Block();
            break;
        case MPI_CYCLE:
            fitMPI_CYCLE();
            break;
        case MPI_MASTER_SLAVE:
            fitMPI_MASTER_SLAVE();
            break;
        case MPI_SLAVE_SLAVE:
            fitMPI_SLAVE_SLAVE();
            break;
        case MPI_PIPELINE:
            fitMPI_PIPELINE();
            break;
        case MPI_SIMD:
            fitMPI_SIMD();
            break;
        case MPI_OMP:
            fitMPI_OMP();
            break;
        case MPI_OMP_SIMD:
            fitMPI_OMP_SIMD();
            break;
        default:
            fitSerial();
            break;
    }
}

/*
 * the function to execute cluster process
 * first if the process ranks 0 initial the centroids and then delivers the data to other processes
 * if the process doesn't rank 0 then it receives the data from process 0
 * then iterate over the loop
 * if the process doesn't rank 0 then it calculates the nearest centroid of each point
 * and changes the cluster labels, last send the cluster labels to process 0
 * if the process ranks 0 it updates the centroids
 */
void KMeansMPI::fitMPI_Block() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
        /*
         * if the process doesn't rank 0 then it receives the data from process 0
         */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
            /*
             * if the process doesn't rank 0 then it receives the centroids from process 0
             */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
            /*
             * if the process ranks 0 then it receives the cluster labels from processes,
             * and then it updates the centroids
             */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

/*
 * the function to execute cluster process
 * first if the process ranks 0 initial the centroids and then delivers the data to other processes
 * if the process doesn't rank 0 then it receives the data from process 0
 * then iterate over the loop
 * if the process doesn't rank 0 then it calculates the nearest centroid of each point
 * and changes the cluster labels, last send the cluster labels to process 0
 * if the process ranks 0 it updates the centroids
 */
void KMeansMPI::fitMPI_CYCLE() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
        /*
         * if the process doesn't rank 0 then it receives the data from process 0
         */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
            /*
             * if the process doesn't rank 0 then it receives the centroids from process 0
             */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
            /*
             * if the process ranks 0 then it receives the cluster labels from processes,
             * and then it updates the centroids
             */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

/*
 * the function to execute cluster process
 * first if the process ranks 0 initial the centroids and then delivers the data to other processes
 * if the process doesn't rank 0 then it receives the data from process 0
 * then iterate over the loop
 * if the process doesn't rank 0 then it calculates the nearest centroid of each point
 * and changes the cluster labels, last send the cluster labels to process 0
 * if the process ranks 0 it updates the centroids
 */
void KMeansMPI::fitMPI_MASTER_SLAVE() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    /*
     * if the process doesn't rank 0 then it receives the data from process 0
     */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        /*
         * if the process doesn't rank 0 then it receives the centroids from process 0
         */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        /*
         * if the process ranks 0 then it receives the cluster labels from processes,
         * and then it updates the centroids
         */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

/*
 * the function to execute cluster process
 * first if the process ranks 0 initial the centroids and then delivers the data to other processes
 * if the process doesn't rank 0 then it receives the data from process 0
 * then iterate over the loop
 * if the process doesn't rank 0 then it only calculates the nearest centroid of each point
 * and changes the cluster labels, last send the cluster labels to process 0
 * if the process ranks 0 it not only calculates the nearest centroid of each point
 * but also updates the centroids
 */
void KMeansMPI::fitMPI_SLAVE_SLAVE() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / size);
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i == size - 1 ? this->N % tasks : tasks;
            MPI_Send(data[i * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
    /*
     * if the process doesn't rank 0 then it receives the data from process 0
     */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[this->rank * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        /*
         * if the process doesn't rank 0 then it receives the centroids from process 0
         */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * whether the process ranks 0 or not it calculates the nearest centroid of each point,
         */
        calculate();
        /*
         * if the process doesn't rank 0 then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
        /*
         * if the process ranks 0 then it receives the cluster labels from processes,
         * and then it updates the centroids
         */
        if (this->rank == 0) {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = source * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

/*
 * the function to execute cluster process
 * first if the process ranks 0 initial the centroids and then delivers the data and centroids  other processes
 * if the process doesn't rank 0 then it receives the data from process 0
 * then iterate over the loop
 * the processes except process 0 calculate the nearest centroid of each point and send the cluster labels to process 0
 * process 0 receives the cluster labels from processes and updates the centroids
 * when it finishes updating one centroid, it sends the centroid to other processes,
 * other processes receive the centroid from process 0 and immediately calculates the nearest centroid of each point
 */
void KMeansMPI::fitMPI_PIPELINE() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data and centroids to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
            MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
        }
    }
    /*
     * if the process doesn't rank 0 then it receives the data and centroids from process 0
     */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        if(this->rank != 0) {
            calculate();
        }
        else{
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_SIMD() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
        /*
         * if the process doesn't rank 0 then it receives the data from process 0
         */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
        /*
         * if the process doesn't rank 0 then it receives the centroids from process 0
         */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
            /*
             * if the process ranks 0 then it receives the cluster labels from processes,
             * and then it updates the centroids
             */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_OMP() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
        /*
         * if the process doesn't rank 0 then it receives the data from process 0
         */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
            /*
             * if the process doesn't rank 0 then it receives the centroids from process 0
             */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
            /*
             * if the process ranks 0 then it receives the cluster labels from processes,
             * and then it updates the centroids
             */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitMPI_OMP_SIMD() {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    this->tasks = ceil(1.0 * this->N / (size - 1));
    /*
     * if the process ranks 0 then it initializes the centroids and then delivers the data to other processes
     */
    if (this->rank == 0) {
        initCentroidsRandom();
        for (int i = 1; i < size; i++) {
            int dataSize = i != this->N / tasks ? tasks : this->N % tasks;
            MPI_Send(data[(i - 1) * this->tasks], dataSize * this->D, MPI_FLOAT, i, DATA_COMM, MPI_COMM_WORLD);
        }
    }
        /*
         * if the process doesn't rank 0 then it receives the data from process 0
         */
    else {
        this->tasks = this->rank == size - 1 ? this->N % tasks : tasks;
        MPI_Recv(data[(this->rank - 1) * this->D], this->tasks * this->D, MPI_FLOAT, 0, DATA_COMM, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    for (int l = 0; l < this->L; l++) {
        /*
         * if process ranks 0 then it delivers centroids to other processes
         */
        if (this->rank == 0) {
            for (int i = 1; i < size; i++) {
                MPI_Send(centroids[0], this->K * this->D, MPI_FLOAT, i, CENTROID_COMM, MPI_COMM_WORLD);
            }
        }
            /*
             * if the process doesn't rank 0 then it receives the centroids from process 0
             */
        else {
            MPI_Recv(centroids[0], this->K * this->D, MPI_FLOAT, 0, CENTROID_COMM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        /*
         * if the process doesn't rank 0 then it calculates the nearest centroid of each point,
         * and then it sends the cluster labels to process 0
         */
        if (this->rank != 0) {
            calculate();
            MPI_Send(this->clusterLabels, this->N, MPI_INT, 0, LABEL_COMM, MPI_COMM_WORLD);
        }
            /*
             * if the process ranks 0 then it receives the cluster labels from processes,
             * and then it updates the centroids
             */
        else {
            int *buff = new int[this->N];
            for (int i = 1; i < size; i++) {
                MPI_Status status;
                MPI_Recv(buff, this->N, MPI_INT, MPI_ANY_SOURCE, LABEL_COMM, MPI_COMM_WORLD, &status);
                int source = status.MPI_SOURCE;
                int begin = (source - 1) * this->tasks;
                int end = source == this->N - 1 ? this->N : begin + this->tasks;
                for (int j = begin; j < end; j++) {
                    this->clusterLabels[j] = buff[j];
                }
            }
            updateCentroids();
        }
    }
}

void KMeansMPI::fitSerial() {
    initCentroidsRandom();
    for (int i = 0; i < this->L; i++) {
        calculate();
        updateCentroids();
    }
}

/*
 * calculate the nearest centroid of each point
 * how to execute calculate() depends on the method
 */
void KMeansMPI::calculate() {
    switch (this->method) {
        case MPI_OMP:
        case MPI_OMP_SIMD:
            calculateMultiThread();
            break;
        case MPI_BLOCK:
        case MPI_CYCLE:
        case MPI_MASTER_SLAVE:
        case MPI_SLAVE_SLAVE:
        case MPI_SIMD:
            calculateSingleThread();
            break;
        case MPI_PIPELINE:
            calculatePipeline();
            break;
        default:
            calculateSerial();
            break;
    }
}

void KMeansMPI::calculateSerial() {
    for (int i = 0; i < this->N; i++) {
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

void KMeansMPI::calculateMultiThread() {
    int begin = this->rank * this->tasks;
    int end = this->rank == size - 1 ? this->N : begin + this->tasks;
    int i, k;
    float min = 1e9;
    int minIndex = 0;
    float dis;
#pragma omp parallel num_threads(this->threadNum) default(none) \
    private(i, k, min, minIndex, dis) \
    shared(this->data, this->clusterLabels, this->centroids, this->N, this->K, begin, end)
    {
#pragma omp for schedule(static)
        {
            for (i = begin; i < end; i++) {
                for (k = 0; k < this->K; k++) {
                    dis = calculateDistance(this->data[i], this->centroids[k]);
                    if (dis < min) {
                        min = dis;
                        minIndex = k;
                    }
                }
                this->clusterLabels[i] = minIndex;
            }
        }
    }
}

void KMeansMPI::calculateSingleThread() {
    int begin = this->method == MPI_SLAVE_SLAVE? this->rank * this->tasks : (this->rank - 1) * this->tasks;
    int end = this->rank == size - 1 ? this->N : begin + this->tasks;
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

void KMeansMPI::calculatePipeline() {

}

float KMeansMPI::calculateDistance(const float *dataItem, const float *centroidItem) const {
    switch (this->method) {
        case MPI_BLOCK:
        case MPI_CYCLE:
        case MPI_PIPELINE:
        case MPI_MASTER_SLAVE:
        case MPI_SLAVE_SLAVE:
        case MPI_OMP:
        default:
            return calculateDistanceSerial(dataItem, centroidItem);
        case MPI_SIMD:
        case MPI_OMP_SIMD:
            return calculateDistanceSIMD(dataItem, centroidItem);
    }
}

float KMeansMPI::calculateDistanceSerial(const float *dataItem, const float *centroidItem) const {
    float dis = 0;
    for (int i = 0; i < this->D; i++)
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    return dis;
}

float KMeansMPI::calculateDistanceSIMD(const float *dataItem, const float *centroidItem) const {
    float dis = 0;
    for (int i = 0; i < this->D - this->D % 4; i += 4) {
        float32x4_t tmpData, centroid;
        tmpData = vmovq_n_f32(&dataItem[i]);
        centroid = vmovq_n_f32(&centroidItem[i]);
        float32x4_t diff = vsubq_f32(tmpData, centroid);
        float32x4_t square = vmulq_f32(diff, diff);
        float sum[4];
        vst1q_f32(sum, square);
        dis += sum[0] + sum[1] + sum[2] + sum[3];
    }
    for (int i = this->D - this->D % 4; i < this->D; i++) {
        dis += (dataItem[i] - centroidItem[i]) * (dataItem[i] - centroidItem[i]);
    }
    return dis;
}

/*
 * update the centroids
 * how to execute updateCentroids() depends on the method
 */
void KMeansMPI::updateCentroids() {
    switch (this->method) {
        case MPI_BLOCK:
        case MPI_CYCLE:
        case MPI_MASTER_SLAVE:
        case MPI_SLAVE_SLAVE:
        default:
            updateCentroidsSerial();
            break;
        case MPI_SIMD:
            updateCentroidsSIMD();
            break;
        case MPI_OMP:
            updateCentroidsOMP();
            break;
        case MPI_OMP_SIMD:
            updateCentroidsOMP_SIMD();
            break;
        case MPI_PIPELINE:
            updateCentroidsPipeline();
            break;
    }
}

void KMeansMPI::updateCentroidsSerial() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster
    for (int i = 0; i < this->N; i++) {
        int cluster = this->clusterLabels[i];
        for (int j = 0; j < this->D; j++) {
            this->centroids[cluster][j] += this->data[i][j];
        }
        this->clusterCount[cluster]++;
    }
    // calculate the mean of the cluster
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D; j++) {
            this->centroids[i][j] /= (float) this->clusterCount[i];
        }
    }
}

void KMeansMPI::updateCentroidsSIMD() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    // accumulate the data of each dimension in the cluster using SSE
    for (int i = 0; i < this->N; i++) {
        int cluster = this->clusterLabels[i];
        this->clusterCount[cluster]++;
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            float32x4_t tmpData = vmovq_n_f32(&this->data[i][j]);
            float32x4_t centroid = vmovq_n_f32(&this->centroids[cluster][j]);
            float32x4_t sum = vaddq_f32(tmpData, centroid);
            vst1q_f32(&this->centroids[cluster][j], sum);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[cluster][j] += this->data[i][j];
        }
    }
    // calculate the mean of the cluster using SSE
    for (int i = 0; i < this->K; i++) {
        for (int j = 0; j < this->D - this->D % 4; j += 4) {
            float32x4_t tmpData = vmovq_n_f32(&this->centroids[i][j]);
            float32x4_t count = vmovq_n_f32(reinterpret_cast<const float *>(&this->clusterCount[i]));
            float32x4_t mean = vdivq_f32(tmpData, count);
            vst1q_f32(&this->centroids[cluster][j], sum);
        }
        for (int j = this->D - this->D % 4; j < this->D; j++) {
            this->centroids[i][j] /= (float) this->clusterCount[i];
        }
    }
}

void KMeansMPI::updateCentroidsOMP() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i, j, cluster;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(static)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                cluster = this->clusterLabels[i];
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[cluster][j] += this->data[i][j];
                    }
                }
                this->clusterCount[cluster]++;
            }
        }
#pragma omp for schedule(static)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
#pragma omp simd
                {
                    for (j = 0; j < this->D; j++) {
                        this->centroids[i][j] /= (float) this->clusterCount[i];
                    }
                }
            }
        }
    }
}

void KMeansMPI::updateCentroidsOMP_SIMD() {
    // initialize the number of each cluster as 0
    memset(this->clusterCount, 0, sizeof(int) * K);
    int i, j, cluster;
    float32x4_t tmpData, centroid, sum, count, mean;
#pragma omp parallel default(none) num_threads(threadNum) \
    private(i, j, cluster, tmpData, centroid, sum, count, mean) \
    shared(this->data, this->clusterLabels, this->clusterLabels, this->clusterCount, this->N, this->K)
    {
#pragma omp for schedule(guided)
        {
            // accumulate the data of each dimension in the cluster
            for (i = 0; i < this->N; i++) {
                int cluster = this->clusterLabels[i];
                this->clusterCount[cluster]++;
                for (j = 0; j < this->D - this->D % 4; j += 4) {
                    float32x4_t tmpData = vmovq_n_f32(&this->data[i][j]);
                    float32x4_t centroid = vmovq_n_f32(&this->centroids[cluster][j]);
                    float32x4_t sum = vaddq_f32(tmpData, centroid);
                    vst1q_f32(&this->centroids[cluster][j], sum);
                }
                for (j = this->D - this->D % 4; j < this->D; j++) {
                    this->centroids[cluster][j] += this->data[i][j];
                }
            }
        }
#pragma omp for schedule(guided)
        {
            // calculate the mean of the cluster
            for (i = 0; i < this->K; i++) {
                for (j = 0; j < this->D - this->D % 4; j += 4) {
                    float32x4_t tmpData = vmovq_n_f32(&this->centroids[i][j]);
                    float32x4_t count = vmovq_n_f32(reinterpret_cast<const float *>(&this->clusterCount[i]));
                    float32x4_t mean = vdivq_f32(tmpData, count);
                    vst1q_f32(&this->centroids[cluster][j], sum);
                }
                for (j = this->D - this->D % 4; j < this->D; j++) {
                    this->centroids[i][j] /= (float) this->clusterCount[i];
                }
            }
        }
    }
}

void KMeansMPI::updateCentroidsPipeline() {

}