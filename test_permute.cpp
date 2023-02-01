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


int main(){
    int N = 8; // number of elements in test array
    int M = 1000; // how many times to repeat
    int R = 1000; // orders

    srand(0);

    vector<datatype> data(N);
    vector<int> order(N);

    vector<datatype> sums(R);
    vector<float> stdvar(M); 

    for(int m = 0; m<M; ++m){
        //init
        for(int i = 0; i<N; ++i){
            order[i] = i;
            data[i] = (datatype) (rand()/(RAND_MAX-1.0)); 
        }

        //random non-repeating permutations
        auto rng = std::default_random_engine {};
        for(int r = 0; r<R; ++r){
            sums[r]=makeSum(data, order, N); 
            std::shuffle(std::begin(order), std::end(order), rng);
        }

        // datatype sInOrder = makeSum(data, order, N);
        // std::reverse(order.begin(), order.end());
        // datatype sRevOrder = makeSum(data, order, N);


        // stdvar[m] = std::abs((float)(sInOrder-sRevOrder)); 

        stdvar[m] = getSTD(sums, R);   
    }

    //calculate mean of variances
    float av = 0;
    for(int m = 0; m<M; ++m){
        av+=stdvar[m];
    }
    av/=M;
    std::cout<<"standard deviation: "<<av<<std::endl;
    return 0;
}
