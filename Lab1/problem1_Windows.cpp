#include <iostream>
#include <windows.h>
using namespace std;
const int N = 10000;
int a[N];
int b[N][N];
int sum[N];
int LOOP = 1;

void init()
{
    for(int i=0;i<N;i++)
        a[i]=i;
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            b[i][j]=i+j;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
        {
            sum[i]=0;
            for(int j=0;j<N;j++)
                sum[i]+=a[j]*b[j][i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"ordinary:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int j=0;j<N;j++)
            for(int i=0;i<N;i++)
                sum[i]+=a[j]*b[j][i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"optimize:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

void unroll()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int j=0;j<N;j+=10)
        {
            int tmp0=0,tmp1=0,tmp2=0,tmp3=0,tmp4=0,tmp5=0,tmp6=0,tmp7=0,tmp8=0,tmp9=0;
            for(int i=0;i<N;i++)
            {
                tmp0+=a[j+0]*b[j+0][i];
                tmp1+=a[j+1]*b[j+1][i];
                tmp2+=a[j+2]*b[j+2][i];
                tmp3+=a[j+3]*b[j+3][i];
                tmp4+=a[j+4]*b[j+4][i];
                tmp5+=a[j+5]*b[j+5][i];
                tmp6+=a[j+6]*b[j+6][i];
                tmp6+=a[j+6]*b[j+6][i];
                tmp7+=a[j+7]*b[j+7][i];
                tmp8+=a[j+8]*b[j+8][i];
                tmp9+=a[j+9]*b[j+9][i];
            }
            sum[j+0]=tmp0;
            sum[j+1]=tmp1;
            sum[j+2]=tmp2;
            sum[j+3]=tmp3;
            sum[j+4]=tmp4;
            sum[j+5]=tmp5;
            sum[j+6]=tmp6;
            sum[j+7]=tmp7;
            sum[j+8]=tmp8;
            sum[j+9]=tmp9;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"unroll:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

int main()
{
    init();
    ordinary();
    optimize();
    unroll();
}