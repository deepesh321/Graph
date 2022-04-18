#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <stdio.h>
#include <string>
#include <string.h>
#include <ctime>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <math.h>

using std::cout;
using std::endl;

__global__ void relax(int N, int MAX_VAL, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < N - 1) { // do index < N - 1 because nth element of I array points to the end of E array
        for (int j = d_in_I[index]; j < d_in_I[index + 1]; j++) {
            int w = d_in_W[j];
            int du = d_out_D[index];
            int dv = d_out_D[d_in_E[j]-1];
            int newDist = du + w;

            if (du == MAX_VAL){
                newDist = MAX_VAL;
            }

            if (newDist < dv) {
                atomicMin(&d_out_Di[d_in_E[j]-1],newDist);

            }
        }
    }
}

__global__ void updateDistance(int N, int *d_in_V, int *d_in_I, int *d_in_E, int *d_in_W, int *d_out_D, int *d_out_Di) {
    unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < N) {

        if (d_out_D[index] > d_out_Di[index]) {
            d_out_D[index] = d_out_Di[index];
        }
        d_out_Di[index] = d_out_D[index];
    }
}

void print(std::vector <int> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
}

void loadVector(const char *filename, std::vector<int> &vec)
{
    std::ifstream input;
    input.open(filename);
    int num;
    while ((input >> num) && input.ignore()) {
        vec.push_back(num);
    }
    input.close();
}

int runBellmanFordOnGPU(const char *file, int blockSize) {

    std::string inputFile=file;
    int BLOCK_SIZE = blockSize;
    int MAX_VAL = std::numeric_limits<int>::max();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cout << "Running Bellman Ford on GPU!" << endl;
    cudaEventRecord(start, 0);

    std::vector<int> V, I, E, W;
    //Load data from files
    loadVector((inputFile + "_V.csv").c_str(), V);
    loadVector((inputFile + "_I.csv").c_str(), I);
    loadVector((inputFile + "_E.csv").c_str(), E);
    loadVector((inputFile + "_W.csv").c_str(), W);

    //print(V);

    int N = I.size();
    int BLOCKS = 1;
    BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cout << "Blocks : " << BLOCKS << " Block size : " << BLOCK_SIZE << endl;

    int *d_in_V;
    int *d_in_I;
    int *d_in_E;
    int *d_in_W;
    int *d_out_D; // Final shortest distance
    int *d_out_Di; // Used in keep track of the distance during one single execution of the kernel

    int D[V.size()];
    int Di[V.size()];

    std::fill_n(D, V.size(), MAX_VAL);
    std::fill_n(Di, V.size(), MAX_VAL);

    D[0] = 0;
    Di[0] = 0;

    //allocate memory
    cudaMalloc((void**) &d_in_V, V.size() *sizeof(int));
    cudaMalloc((void**) &d_in_I, I.size() *sizeof(int));
    cudaMalloc((void**) &d_in_E, E.size() *sizeof(int));
    cudaMalloc((void**) &d_in_W, W.size() *sizeof(int));

    cudaMalloc((void**) &d_out_D, V.size() *sizeof(int));
    cudaMalloc((void**) &d_out_Di, V.size() *sizeof(int));
    cudaMemcpy(d_out_D, D, sizeof(int)*V.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_Di, Di, sizeof(int)*V.size(), cudaMemcpyHostToDevice);

    //copy to device memory
    cudaMemcpy(d_in_V, V.data(), V.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_I, I.data(), I.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_E, E.data(), E.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_W, W.data(), W.size() *sizeof(int), cudaMemcpyHostToDevice);

    // Bellman ford
    for (int round = 1; round < V.size(); round++) {
        relax<<<BLOCKS, BLOCK_SIZE>>>(N, MAX_VAL, d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
        updateDistance<<<BLOCKS, BLOCK_SIZE>>>(V.size(), d_in_V, d_in_I, d_in_E, d_in_W, d_out_D, d_out_Di);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cout << "Completed Bellman Ford on GPU!" << endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    int *out_weights = new int[V.size()];

    cudaMemcpy(out_weights, d_out_D, V.size()*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "** average time elapsed : " << elapsedTime << " milli seconds** " << endl;

    //for (int n=0; n<V.size(); n++) 
    //{ 
    //cout << out_weights[n] << " "; 
    //} 

    free(out_weights);
    cudaFree(d_in_V);
    cudaFree(d_in_I);
    cudaFree(d_in_E);
    cudaFree(d_in_W);
    cudaFree(d_out_D);
    cudaFree(d_out_Di);
    return 0;
}

int main(int argc, char **argv) {
    
    std::string file;
    file = argv[1];

    int BLOCK_SIZE = 512;

    runBellmanFordOnGPU(file.c_str(), BLOCK_SIZE);
}