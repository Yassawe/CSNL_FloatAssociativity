#include <floatx.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>  
using namespace std;


mt19937 RNG_distro(20214229);
mt19937 RNG_order(20214229);

// bfloat16
// typedef flx::floatx<8,7> datatype;

// float16
typedef flx::floatx<5,10> datatype;


double cosineSimilarity(const vector<datatype>& v1, const vector<datatype>& v2, int M) {
    double dotProduct = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < M; ++i) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dotProduct / (sqrt(norm1) * sqrt(norm2)); 
}

double averagePairwiseCosineSimilarity(const vector<vector<datatype>>& sums, int R, int M) {
    double sumSimilarities = 0.0;
    int count = 0;
    for (int i = 0; i < R; ++i) {
        for (int j = i + 1; j < R; ++j) {
            double similarity = cosineSimilarity(sums[i], sums[j], M);
            sumSimilarities += similarity;
            count++;
        }
    }
    return sumSimilarities / count;
}

vector<datatype> makeSum(vector<vector<datatype>>& data, vector<int>& order, int N, int M){

    vector<datatype> sum(M);
    int idx;

    for(int elem = 0; elem<M; ++elem){
        sum[elem]=0;

        for(int i = 0; i<N; ++i){
            idx = order[i];
            sum[elem]+=data[idx][elem];

        }
    }

    return sum;
}

void generateRandomNumbers(vector<vector<datatype>>& data, int N, int M){

    float mean = 0.00000028992896587; //from ResNet50 profiling
    float std = 0.000511206; 

    //TODO: Laplace Distribution

    // uniform_real_distribution<float> distro(-10, 10);
    normal_distribution<float> distro(mean, std);

    // KEEP IN MIND: GRADS THAT ARE OF THE SAME IDX, ARE CLOSE IN REAL ML WORKLOADS, HERE IT IS NOT THE CASE. DOES IT MATTER?
    for(int i = 0; i<N; ++i){
        for (int j = 0; j<M; ++j){
            data[i][j] = (datatype) distro(RNG_distro);
        }
    }
}


void runExperiment(int N, int M, int R){

    vector<vector<datatype>> data(N, vector<datatype>(M));
    vector<vector<datatype>> sums(R, vector<datatype> (M));
    
    vector<int> order(N);
    for (int device = 0; device<N; ++device){
        order[device]=device;
    }

    generateRandomNumbers(data, N, M);

    for (int r = 0; r<R; ++r){
        sums[r] = makeSum(data, order, N, M);
        shuffle(order.begin(), order.end(), RNG_order); // probability of repeating is very low, but not 0. ok for now.
    }

    double res = averagePairwiseCosineSimilarity(sums, R, M);

    cout<<"N = " << N << endl;
    cout<<"Average Pairwise Cosine Disimilarity = " << 1.0-res << endl;
    // getting weird results, how can it be larger than 1??
}


int main(){
    
    int Ns[8] = {4, 8, 16, 64, 256, 512, 1024, 4096};
    
    int M = 1000;
    int R = 20;

    cout<<"FLOAT16"<<endl;

    for(int i = 0; i<8; ++i){
        runExperiment(Ns[i], M, R);
    }

    return 0;
}
