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
    init_data();
    for(int i = 0; i < N; i++,cout<<endl)
        for(int j = 0; j < N; j++)
            cout<<&matrix[i][j] << " ";
}