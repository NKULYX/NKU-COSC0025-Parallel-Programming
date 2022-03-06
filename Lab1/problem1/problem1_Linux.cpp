#include <iostream>
#include <sys/time.h>
using namespace std;

#define ull unsigned long long int 

const int N = 80;
ull a[N];
ull b[N][N];
ull sum[N];
int LOOP = 1000;

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
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                sum[i]+=a[j]*b[j][i];
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
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

void unroll()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
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
    gettimeofday(&end,NULL);
    cout<<"unroll:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

int main()
{
    init();
    ordinary();
    optimize();
    unroll();
}