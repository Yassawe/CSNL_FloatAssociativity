#include <floatx.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>    

// bfloat16
typedef flx::floatx<8,7> datatype;

// float16
// typedef flx::floatx<5,10> datatype;

using namespace std;


datatype makeSum(vector<datatype>& data, vector<int>& order, int N){
    datatype sum = 0;
    for(int i = 0; i<N; ++i){
        sum+=data[order[i]];
    }
    return sum/N;
}

float getSTD(vector<datatype> &sums, int R){
    float mean = 0;
    for(int r = 0; r<R; ++r){
        mean+=(float)sums[r];
    }
    mean/=R;
    float st = 0;
    for(int r = 0; r<R; ++r){
        st+= pow((float)(sums[r]-mean),2);
    }

    return sqrt(st)/sqrt(R); 
}


void runExperiment(int N, int M, int R){

    std::mt19937 gen(time(NULL)); 
    std::uniform_real_distribution<> dis(-3, 3);


    vector<datatype> data(N);
    vector<int> order(N);

    vector<datatype> sums(R);

    float avSTD = 0;
    float maxSTD = 0; 
    float avRange = 0;
    float maxRange = 0;

    for(int m = 0; m<M; ++m){
        //init
        for(int i = 0; i<N; ++i){
            order[i] = i;
            data[i] = (datatype) dis(gen); 
        }

        float minV = INFINITY; 
        float maxV = -INFINITY;
        float stdev =  0;
        float r = 0;
        //random non-repeating permutations
        std::mt19937 rng(time(NULL));
        for(int r = 0; r<R; ++r){
            sums[r]=makeSum(data, order, N); 
            shuffle(begin(order), end(order), rng);
            if (sums[r]<minV) minV = sums[r];
            if (sums[r]>maxV) maxV = sums[r];

            //cout<<std::setprecision(20)<<sums[r]<<endl; 
        }

        stdev = getSTD(sums, R);
        r = (float)(maxV-minV);

        avSTD+=stdev;
        if (stdev>maxSTD) maxSTD=stdev;

        avRange+=r;
        if (r>maxRange) maxRange=r;  
    }

    avSTD/=M;
    avRange/=M;

    cout<<"\nN = "<<N<<endl;
    cout<<"Average Standard Deviation = "<<avSTD<<endl;
    cout<<"Maximum Standard Deviation = " <<maxSTD<<endl;
    cout<<"Average range of sum = "<<avRange<<endl;
    cout<<"Maximum range of sum = "<<maxRange<<endl;
    cout<<"\n"<<endl;
}


int main(){
    
    int Ns[7] = {16, 64, 256, 512, 1024, 4096, 16384};
    int M = 100;
    int R = 1000;

    cout<<"BFLOAT16"<<endl;

    for(int i = 0; i<7; ++i){
        runExperiment(Ns[i], M, R);
    }

    return 0;
}
