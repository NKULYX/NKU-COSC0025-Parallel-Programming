#include <iostream>
#include <sys/time.h>
using namespace std;

#define ull unsigned long long int

const ull N = 33554432;
ull a[N];
int LOOP = 1;

void init()
{
    for (ull i = 0; i < N; i++)
        a[i] = i;
}

void ordinary()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        // init();
        ull sum = 0;
        for (int i = 0; i < N; i++)
            sum += a[i];
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

int main()
{
    init();
    ordinary();
}