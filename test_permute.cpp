#include <floatx.hpp>
#include <vector>


// bfloat16
// typedef flx::floatx<8,7> datatype;

// float16

typedef flx::floatx<5,11> datatype;

using namespace std;

int main(){
    int N = 64;
    int M = 1000;
    vector<vector<datatype>> data(M, vector<datatype>(N)); // MxN array

    
    
    



    return 0;
}