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
    vector_t guess = generate_random_guess(N);
    pair<double, vector_t> res = dominant_eigenvalue(M, guess, niter);
    // Output results to stdout
    cout << res.first << endl;
    for (int i = 0; i < N; ++i){
        cout << res.second[i] << endl;
    }
    return 0;
}
