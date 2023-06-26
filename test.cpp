#include <floatx.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <iomanip>  
using namespace std;


mt19937 RNG_distro(2220940);
mt19937 RNG_order(20214229);

// bfloat16
// typedef flx::floatx<8,7> datatype;

// float16
typedef flx::floatx<5,10> datatype;


// double euclideanDistanceNorm(const vector<datatype>& v1, const vector<datatype>& v2){

// }

double cosineSimilarity(const vector<datatype>& v1, const vector<datatype>& v2) {
    double dotProduct = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dotProduct / (sqrt(norm1) * sqrt(norm2)); 
}

double averagePairwiseSimilarity(const vector<vector<datatype>>& vectors, double (*distanceFunction)(const vector<datatype>& v1, const vector<datatype>& v2)) {
    double sumSimilarities = 0.0;
    int count = 0;
    for (int i = 0; i < vectors.size(); ++i) {
        for (int j = i + 1; j < vectors.size(); ++j) {
            double similarity = distanceFunction(vectors[i], vectors[j]);
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

    // uniform_real_distribution<float> distro(-1, 1);
    normal_distribution<float> distro(mean, std);

    // KEEP IN MIND: GRADS THAT ARE OF THE SAME IDX, ARE CLOSE IN REAL ML WORKLOADS, HERE IT IS NOT THE CASE. DOES IT MATTER?
    for(int i = 0; i<N; ++i){
        for (int j = 0; j<M; ++j){
            data[i][j] = (datatype) distro(RNG_distro);
        }
    }
}


void runExperiment_Hier_2stage(int N, int M, int R){

    vector<vector<datatype>> data(N, vector<datatype>(M));
    vector<vector<datatype>> sums(R, vector<datatype> (M));

    int a = round(log2(N));
    int st1_size = pow(2, a - round(a/2));
    int st2_size = pow(2, round(a/2));

    vector<vector<datatype>> stage2(st2_size, vector<datatype>(M));
    vector<int> st2_order(st2_size);
    iota(st2_order.begin(), st2_order.end(), 0);

    // cout<<st1_size<<endl;
    // cout<<st2_size<<endl;

    vector<int> order(N);
    for (int device = 0; device<N; ++device){
        order[device]=device;
    }

    generateRandomNumbers(data, N, M);

    for(int r=0; r<R; ++r){
        for(int st2idx = 0; st2idx<st2_size; ++st2idx){
            vector<int> st1_order = vector<int>(order.begin() + st1_size*st2idx, order.begin() + st1_size*(st2idx+1));
            stage2[st2idx] = makeSum(data, st1_order, st1_size, M);
        }

        sums[r] = makeSum(stage2, st2_order, st2_size, M);
        shuffle(order.begin(), order.end(), RNG_order); 
    }

    double res = averagePairwiseSimilarity(sums, &cosineSimilarity);
    cout<<"N = " << N << endl;
    cout<<"Average Pairwise Cosine Disimilarity = " << 1.0-res << endl;
}

void runExperiment_Hier_3stage(int N, int M, int R){
    vector<vector<datatype>> data(N, vector<datatype>(M));
    vector<vector<datatype>> sums(R, vector<datatype> (M));

    int a = round(log2(N));
    int st1_size = pow(2, a-round(a/3)-round(a/3));
    int st2_size = pow(2, round(a/3));
    int st3_size = pow(2, round(a/3));

    // cout<<st1_size<<endl;
    // cout<<st2_size<<endl;
    // cout<<st3_size<<endl;

    vector<vector<datatype>> intermediate(st2_size*st3_size, vector<datatype>(M));
    vector<int> inter_order(st2_size*st3_size);
    iota(inter_order.begin(), inter_order.end(), 0);


    vector<vector<datatype>> stage3(st3_size, vector<datatype>(M));
    vector<int> st3_order(st3_size);
    iota(st3_order.begin(), st3_order.end(), 0);


    vector<int> order(N);
    for (int device = 0; device<N; ++device){
        order[device]=device;
    }

    generateRandomNumbers(data, N, M);

    for(int r=0; r<R; ++r){
        for(int idx = 0; idx<st2_size*st3_size; ++idx){
            vector<int> st1_order = vector<int>(order.begin() + st1_size*idx, order.begin() + st1_size*(idx+1));
            intermediate[idx] = makeSum(data, st1_order, st1_size, M);
        }

        for(int st3_idx=0; st3_idx<st3_size; ++st3_idx){
            vector<int> st2_order = vector<int>(inter_order.begin()+st2_size*st3_idx, inter_order.begin()+st2_size*(st3_idx+1));
            stage3[st3_idx] = makeSum(intermediate, st2_order, st2_size, M);
        }


        sums[r] = makeSum(stage3, st3_order, st3_size, M);
        shuffle(order.begin(), order.end(), RNG_order); 
    }


    double res = averagePairwiseSimilarity(sums, &cosineSimilarity);
    cout<<"N = " << N << endl;
    cout<<"Average Pairwise Cosine Disimilarity = " << 1.0-res << endl;

}


void runExperiment_flat(int N, int M, int R){

    vector<vector<datatype>> data(N, vector<datatype>(M));
    vector<vector<datatype>> sums(R, vector<datatype> (M));
    
    vector<int> order(N);
    for (int device = 0; device<N; ++device){
        order[device]=device;
    }

    generateRandomNumbers(data, N, M);

    for (int r = 0; r<R; ++r){
        sums[r] = makeSum(data, order, N, M);
        //cout<<sums[r][12]<<endl;
        shuffle(order.begin(), order.end(), RNG_order); // probability of repeating is very low, but not 0. keep that in mind.
    }

    double res = averagePairwiseSimilarity(sums, &cosineSimilarity);

    cout<<"N = " << N << endl;
    cout<<"Average Pairwise Cosine Disimilarity = " << 1.0-res << endl;
    // getting weird results, how can it be larger than 1??
}


int main(){
    
    int num = 11;

    int Ns[num] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    
    int M = 1000;
    int R = 100;

    cout<<"3 STAGE"<<endl;
    cout<<"FLOAT16"<<endl;
    

    for(int i = 1; i<num; ++i){
        runExperiment_flat(Ns[i], M, R);
        runExperiment_Hier_2stage(Ns[i], M, R);
        runExperiment_Hier_3stage(Ns[i], M, R);
    }

    return 0;
}
