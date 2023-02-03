#include <floatx.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>

// bfloat16
typedef flx::floatx<8,7> datatype;

// float16
// typedef flx::floatx<5,11> datatype;

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
        st+=(sums[r]-mean)*(sums[r]-mean);
    }

    return sqrt(st)/sqrt(R);   
}


void runExperiment(int N, int M, int R){
    
    //srand(20214229);

    vector<datatype> data(N);
    vector<int> order(N);

    vector<datatype> sums(R);
    vector<float> stdvar(M); 
    vector<float> ranges(M);

    for(int m = 0; m<M; ++m){
        //init
        for(int i = 0; i<N; ++i){
            order[i] = i;
            data[i] = (datatype) (rand()/(RAND_MAX-1.0)); 
        }

        float minV = 1; 
        float maxV = 0;

        //random non-repeating permutations
        auto rng = default_random_engine {};
        for(int r = 0; r<R; ++r){
            sums[r]=makeSum(data, order, N); 
            shuffle(begin(order), end(order), rng);
            if (sums[r]<minV) minV = sums[r];
            if (sums[r]>maxV) maxV = sums[r]; 
        }
 
        stdvar[m] = getSTD(sums, R);  
        ranges[m] = (float)(maxV-minV); 
    }

    //calculate mean of variances
    float avSTD = 0;
    float maxSTD = stdvar[0]; 
    float avRange = 0;
    float maxRange = ranges[0];

    for(int m = 0; m<M; ++m){
        avSTD+=stdvar[m];
        avRange+=ranges[m];

        if (stdvar[m]>maxSTD) maxSTD = stdvar[m];
        if (ranges[m]>maxRange) maxRange = ranges[m];
    }
    avSTD/=M;
    avRange/=M;

    cout<<"\nN = "<<N<<endl;
    cout<<"Average Standard Deviation = "<<avSTD<<endl;
    cout<<"Maximum Standard Deviation = " <<maxSTD<<endl;
    cout<<"Average range of sum = "<<avRange<<endl;
    cout<<"Maximum range of sum = "<<maxRange<<endl;
}


int main(){
    
    int Ns[6] = {16, 64, 256, 512, 1024, 4096};
    int M = 1000;
    int R = 1000;

    cout<<"BFLOAT16"<<endl;

    for(int i = 0; i<6; ++i){
        runExperiment(Ns[i], M, R);
    }

    return 0;
}
