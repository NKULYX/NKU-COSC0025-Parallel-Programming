#include <iostream>
#include <windows.h>
using namespace std;

const int N = 100;
const int L = 100;
float matrix[N][N];

void init();
void calculate_serial();

int main()
{
    init();
    calculate_serial();
    system("pause");
}

void init()
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            matrix[i][j] = rand() * 1.0 / RAND_MAX * L;
    for(int i = 0; i < N - 1; i++)
        for(int j = i + 1; j < N; j++)
            for(int k = 0; k < N; k++)
                matrix[j][k] += matrix[i][k];
}

void calculate_serial()
{
    for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		}
		matrix[k][k] = 1;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			matrix[i][k] = 0;
		}
	}
}