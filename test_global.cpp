#include <floatx.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>  
#include <boost/math/distributions/laplace.hpp>





// bfloat16
// typedef flx::floatx<8,7> datatype;

// float16
typedef flx::floatx<5,10> datatype;

using namespace std;


datatype cosineSimilarity(){
    //TODO
}

datatype averagePairwiseCosineSimilarity(){
    //TODO
}


datatype makeSum(vector<datatype>& data, vector<int>& order, int N){
    datatype sum = 0;
    for(int i = 0; i<N; ++i){
        sum+=data[order[i]];
    }
    return sum;
}

void generateRandomNumbers(vector<vector<datatype>>& data){

    float mean = 0.00000028992896587; //from ResNet50 profiling
    float std = 0.000511206; 


    //TODO: Laplace Distribution

    // uniform_real_distribution<datatype> dis(-1, 1);
    normal_distribution<datatype> dis(mean, std);




    // KEEP IN MIND: VALS THAT ARE OF THE SAME IDX, ARE CLOSE IN REAL ML WORKLOADS, HERE IT IS NOT THE CASE. DOES IT MATTER?
    for(int i = 0; i<N; ++i){
        for (int j = 0; j<M; ++j){
            data[i][j] = 
        }
    }
}


void runExperiment(int N, int M, int R){

    vector<vector<datatype>> data(N, vector<datatype>(M));

    vector<int> order(N);

    vector<datatype> sums(R);

    // generate N columns of M random numbers

    for(int i = 0; i<N; ++i){
        order[i] = i;
        data[i] = (datatype) dis(gen); 
    }

    for(int r = 0; r<R; ++r){


    }

    for(int m = 0; m<M; ++m){
        
        

        std::mt19937 rng(time(NULL));
        for(int r = 0; r<R; ++r){
            sums[r]=makeSum(data, order, N); 
            shuffle(begin(order), end(order), rng);
        }
    }

    
}


int main(){
    
    int Ns[7] = {4, 8, 16, 64, 256, 512, 1024};
    
    int M = 1;
    int R = 100;

    cout<<"BFLOAT16"<<endl;

    for(int i = 0; i<7; ++i){
        runExperiment(Ns[i], M, R);
    }

    // runExperiment(16, 1, 24);
    return 0;
}



// float getSTD(vector<datatype> &sums, int R){
//     float mean = 0;
//     for(int r = 0; r<R; ++r){
//         mean+=(float)sums[r];
//     }
//     mean/=R;
//     float st = 0;
//     for(int r = 0; r<R; ++r){
//         st+= pow((float)(sums[r]-mean),2);
//     }

//     return sqrt(st)/sqrt(R); 
// }