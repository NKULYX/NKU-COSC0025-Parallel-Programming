#include <iostream>
#include <sys/time.h>
using namespace std;

#define ull unsigned long long int 

const int N = 3000;
ull a[N];
ull b[N][N];
ull sum[N];
int LOOP = 1;

void init()
{
    for(int i=0;i<N;i++)
        a[i]=i;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            b[i][j]=i+j;
}

void optimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int j=0;j<N;j++)
            for(int i=0;i<N;i++)
                sum[i]+=b[j][i]*a[j];
    }
    gettimeofday(&end,NULL);
    cout<<"optimize:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

int main()
{
    init();
    optimize();
}