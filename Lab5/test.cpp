//
// Created by Lenovo on 2022/6/1.
//
#include <iostream>
using namespace std;

int N = 5;
const int L = 100;
const int LOOP = 1;
float **data;
float **matrix = nullptr;

void init_data()
{
    data = new float *[N], matrix = new float *[N];
    float * tmp = new float[N*N];
    for (int i = 0; i < N; i++)
        data[i] = new float[N],
        matrix[i] = tmp+i*N;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            data[i][j] = rand() * 1.0 / RAND_MAX * L;
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            for (int k = 0; k < N; k++)
                data[j][k] += data[i][k];
}

int main()
{
    int **a, **b;
    a = new int *[2];
    b = new int *[2];
    a[0] = new int[2];
    a[1] = new int[2];
    b[0] = new int[2];
    b[1] = new int[2];
    a[0][0] = 1;
    b[0][0] = a[0][0];
    a[0][0] = 2;
    cout<<b[0][0]<<endl;
}