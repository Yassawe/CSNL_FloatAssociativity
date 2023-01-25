#include "half.hpp"
#include <floatx.hpp>



//bfloat16
typedef flx::floatx<8,7> datatype;

//float16
// typedef flx::floatx<5,11> datatype;





void stats(float* difference, int size){
    float max = difference[0];
    float sum = 0;
    int nonzeros = 0;
    for (int i = 1; i < size; i++) {
        sum += difference[i];
        
        if (difference[i] > max) {
            max = difference[i];
        }

        if (difference[i]!=0) {
          nonzeros++;
        }

    }
    
    float meanWithZeros = sum / size;
    float meanWithoutZeros = sum/nonzeros;
    
    printf("\nElements that are different: %.3f percent \n", ((float)nonzeros/(float)size)*100);
    printf("Maximum difference: %.10f\n", max);
    printf("Mean difference: %.10f\n\n", meanWithZeros);
    // printf("Mean difference (not including zeros): %.16f\n\n", meanWithoutZeros);
}


int main(int argc, char* argv[]){
    // printf("Hello from TensorFlow C library version %s\n", TF_Version());

    int N = 16384;
    int size = 1000;
    int G = 64;
    int numGroups = N/G;

    // generate random nums
    datatype** devices = (datatype **) malloc(N*sizeof(datatype *));
    datatype* sum_together = (datatype*) malloc(size*sizeof(datatype));
    datatype* sum_hier = (datatype*) malloc(size*sizeof(datatype));
    

    for(int n = 0; n<N; ++n){
        srand(20214229*n);
        devices[n] = (datatype *) malloc(size*sizeof(datatype)); 
        for(int i = 0; i<size; ++i){
            devices[n][i] = (datatype) (rand()/(RAND_MAX-1.0));
        }
    }

    // together
    for(int i = 0; i<size; ++i){
        sum_together[i] = 0;
        for(int n =0; n<N; ++n){
            sum_together[i]+=devices[n][i];
        }
        // printf("\n%f\n", (float) sum_together[i]);
        sum_together[i]=sum_together[i]/N;
    }

    //group by group
    for(int i = 0; i<size; ++i){
        sum_hier[i] = 0;
        for(int g = 0; g<numGroups; ++g){
            datatype partialSum = (datatype) 0.0;
            for(int m=0; m<G; ++m){
                partialSum+=devices[g*G+m][i];
            }
            sum_hier[i]+=partialSum;
        }
        sum_hier[i]=sum_hier[i]/N;
        // printf("\n%f\n", (float) sum_hier[i]);
    }

    //analyze
    float difference[size];
    for(int i=0; i<size; ++i){
        difference[i] = fabs(sum_together[i] - sum_hier[i]); 
    }
    stats(difference, size);
    return 0;
}