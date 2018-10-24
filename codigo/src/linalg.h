#include <cmath>
#include <vector>
#include <random>
#include "../src_catedra/types.h"
using namespace std;

// Some type definitions
typedef vector<double> vector_t;
typedef vector<vector_t> matrix;
// Random number generation
std::default_random_engine generator;
//std::normal_distribution<double> distribution(0, 1);
std::uniform_real_distribution<double> distribution(0.0,10.0);

vector_t dot(matrix& A, vector_t& x){
	vector_t res = vector_t(A.size());
	for (int i = 0; i < A.size(); ++i){
    	for (int j = 0; j < x.size(); ++j){
    		res[i] += A[i][j] * x[j];
    	}
    }
    return res;
}

double transpose_dot(vector_t& v, vector_t& w){
	double res = 0;
	for (int i = 0; i < v.size(); ++i){
		res += v[i] * w[i];
	}
	return res;
}

matrix transpose(matrix& A){
	matrix res = matrix(A[0].size(), vector_t(A.size(), 0));
	for (int i = 0; i < res.size(); ++i){
		for (int j = 0; j < res[0].size(); ++j){
			res[i][j] = A[j][i];
		}
	}
	return res;
}

double norm(vector_t& v){
	double norm_2 = 0;
	for (int i = 0; i < v.size(); ++i){
		norm_2 += v[i] * v[i];
	}
	norm_2 = sqrt(norm_2);
	return norm_2;
}

double norm_2_distance(vector_t& x, vector_t& y){
	vector_t sub = vector_t(x.size(), 0);
	for (int i = 0; i < x.size(); ++i){
		sub[i] = x[i] - y[i];
	}
    return norm(sub);
}

void normalize(vector_t& v){
	double norm_2 = norm(v);
	for (int i = 0; i < v.size(); ++i) v[i] /= norm_2;
}

vector_t sample_mean(matrix& samples){
	int N = samples.size();
	int M = samples[0].size();
	vector_t mu = vector_t(M, 0);
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			mu[j] += samples[i][j] / N;
		}
	}
	return mu;
}

matrix covariance_matrix(matrix& samples){
	int N = samples.size();
	int M = samples[0].size();
	vector_t mu = sample_mean(samples);
	matrix X_T = matrix(M, vector_t(N, 0));
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			X_T[j][i] = (samples[i][j] - mu[j]) / sqrt(N - 1);
		}
	}
	
	std::cerr << "Building Covaraince matrix  " <<'\r';
	matrix Cov = matrix(M, vector_t(M, 0));
	for (int i = 0; i < M; ++i){
		std::cerr << "Computing row: " << i << '/' << M <<"          \r";
		for (int j = 0; j < M; ++j){
			for (int k = 0; k < N; ++k){
				// Using X_T won't trash the cache.
				Cov[i][j] += X_T[i][k] * X_T[j][k];
			}
		}
	}
	return Cov;
}

pair<double, vector_t> dominant_eigenvalue(matrix& M, vector_t guess, int niter){
	vector_t v = guess;
	for (int i = 0; i < niter; ++i){
		v = dot(M, v);
		normalize(v);
	}
	vector_t Mv = dot(M, v);
	double lambda = transpose_dot(v, Mv) / (pow(norm(v), 2));
	return make_pair(lambda, v);
}

void deflate(matrix& M, vector_t& v, double lambda){
	for (int i = 0; i < M.size(); ++i){
		for (int j = 0; j < M.size(); ++j){
			M[i][j] -= lambda * v[i] * v[j];
		}
	}
}

vector_t generate_random_guess(int N){
	/*
	 * Generate a random vector by sampling each
	 * component from a standard normal distribution.
	 */
	vector_t res = vector_t(N, 0);
	for (int i = 0; i < N; ++i) res[i] = distribution(generator);
	return res;
}

void print_matrix(matrix& A){
	std::cout << "---------" << std::endl;
	for (int i = 0; i < A.size(); ++i){
		std::cout << '[';
		for (int j = 0; j < A[0].size(); ++j){
			std::cout << A[i][j] << ',';
		}
		std::cout << ']' << std::endl;
	}
	std::cout << "---------" << std::endl;
}

void print_vector(vector_t& v){
	std::cout << "---------" << std::endl;
	std::cout << "[";
	for (int i = 0; i < v.size(); ++i){
		std::cout << v[i] << ", ";
	}
	std::cout << "]" << std::endl;
	std::cout << "---------" << std::endl;
}

matrix read_matrix_from_file(string fname){
    std::string line;
    std::ifstream infile;
    infile.open(fname);
    cout << fname << endl;
    if (infile.fail()) throw std::runtime_error("Ocurrió un error al abrir el archivo.");
    std::getline(infile, line);
    int N = stoi(line);

    matrix res = matrix(N, vector_t(N, 0));
    for (int i = 0; i < N; ++i){
        std::getline(infile, line);
        std::istringstream linestream(line);
        for (int j = 0; j < N; ++j){
            std::string value;
            getline(linestream, value, ',');
            res[i][j] = std::stod(value);
        }
    }
    infile.close();
    return res;
}

void write_matrix_to_file(matrix& M, string fname){
    std::ofstream outfile;
    outfile.open(fname);
    cout << "Writing matrix to file: "<< fname << endl;
    if (outfile.fail()) throw std::runtime_error("Ocurrió un error al abrir el archivo.");
    outfile << M.size() << "," << M[0].size() << endl;
    for (int i = 0; i < M.size(); ++i){
        for (int j = 0; j < M[0].size(); ++j){
            outfile << M[i][j] << ',';
        }
       	outfile << ',';
    }
    outfile.close();
}


void write_vector_to_file(vector_t& v, string fname){
    std::ofstream outfile;
    outfile.open(fname);
    cout << "Writing matrix to file: "<< fname << endl;
    if (outfile.fail()) throw std::runtime_error("Ocurrió un error al abrir el archivo.");
    outfile << v.size() << endl;
    for (int i = 0; i < v.size(); ++i){
        outfile << v[i] << endl;
    }
    outfile.close();
}
