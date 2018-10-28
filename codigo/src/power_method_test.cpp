#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <string>
#include <stdexcept>
#include <stdlib.h>

#include <chrono>

#include "linalg.h"
using namespace std;


int main(int argc, char * argv[]){
    string input_file = string(argv[1]);
    int niter = stoi(argv[2], NULL);
    int N;
    // Read matrix from file
    matrix M = read_matrix_from_file(input_file);
    N= M.size();
    // Generate random guess and compute eigenvector/eigenvalue
    auto start = chrono::system_clock::now();
    vector_t guess = generate_random_guess(N);
    pair<double, vector_t> res = dominant_eigenvalue(M, guess, niter, 0);
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end-start;
    // Output results to stdout
    cout << N << endl;
    cout << elapsed_seconds.count() << endl;
    cout << res.first << endl;
    for (int i = 0; i < N; ++i){
        cout << res.second[i] << endl;
    }
    return 0;
}
