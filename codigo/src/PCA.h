#include <cmath>
#include <vector>
#include <random>
#include "../src_catedra/types.h"

using namespace std;

std::default_random_engine generator;
std::normal_distribution<double> distribution(0, 1);

typedef vector<double> vector_t;
typedef vector<vector_t> matrix;

void print_matrix(matrix& A){
	cout << "---------" << endl;
	for (int i = 0; i < A.size(); ++i){
		cout << '[';
		for (int j = 0; j < A[0].size(); ++j){
			cout << A[i][j] << ',';
		}
		cout << ']' << endl;
	}
	cout << "---------" << endl;
}

void print_vector(vector_t& v){
	cout << "---------" << endl;
	cout << "[";
	for (int i = 0; i < v.size(); ++i){
		cout << v[i] << ", ";
	}
	cout << "]" << endl;
	cout << "---------" << endl;
}

class PCA {
public:
	PCA();
	void fit(VectorizedEntriesMap& dataset);
	vector_t transform(vector_t& x, int alpha);
	VectorizedEntry transform(VectorizedEntry& x, int alpha);
private:
	matrix eigenbasis;
	vector_t eigenvalues;
	
};

PCA::PCA(){
}

vector_t dot(matrix& A, vector_t& x){
	vector_t res = vector_t(A[0].size());
	for (int i = 0; i < A.size(); ++i){
    	for (int j = 0; j < A[0].size(); ++j){
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
	matrix X = matrix(N, vector_t(M, 0));
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			X[i][j] = (samples[i][j] - mu[j]) / sqrt(N - 1);
		}
	}
	matrix X_T = matrix(M, vector_t(N, 0));
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < M; ++j){
			X_T[j][i] = X[i][j];
		}
	}
	cout << "Building Cov" << endl;
	matrix Cov = matrix(M, vector_t(M, 0));
	for (int i = 0; i < M; ++i){
		std::cerr << "Computing row: " << i << '/' << M <<'\r';
		for (int j = 0; j < M; ++j){
			for (int k = 0; k < N; ++k){
				// Using X_T won't trash the cache.
				Cov[i][j] += X_T[i][k] * X_T[j][k];
			}
		}
	}
	return Cov;
}

pair<double, vector_t> dominant_eigenvector(matrix& M, vector_t guess, int niter){
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
	vector_t res = vector_t(N, 0);
	for (int i = 0; i < N; ++i){
		res[i] = distribution(generator);
	}
	return res;
}


void PCA::fit(VectorizedEntriesMap& dataset){
	matrix X = matrix(
		dataset.size(),
		vector_t(dataset.begin()->second.bag_of_words.size(), 0));
	int i = 0;
	for (auto sample = dataset.begin(); sample != dataset.end(); ++sample){
		X[i] = sample->second.bag_of_words;
		i++;
	}
	/* Build samples covariance matrix */
	std::cerr << "Building covariance matrix..." << '\r';
	matrix M = covariance_matrix(X);
	/* Eigenvector and eigenvalues matrices */
	matrix P = matrix(M.size(), vector_t(M.size(), 0)); // Eigenvector matrix.
	vector_t D = vector_t(M.size(), 0); // Eigenvalues.
	std::cerr << "Computing eigenvector basis" << '\r';
	for (int i = 0; i < M.size(); ++i){
		std::cerr << "Computing eigenvector: " << i << '/' << M.size() <<'\r';
		/*
		 * Compute dominant eigenvalue and its correspondent eigenvector
		 * and deflate the covariance matrix.
		 */
		vector_t v_i = generate_random_guess(M.size());
		pair<double, vector_t> component = dominant_eigenvector(M, v_i, 20);
		D[i] = component.first;
		P[i] = component.second;
		deflate(M, component.second, component.first);
	}
	/* Set up the basis matrix */
	this->eigenbasis = P;
	this->eigenvalues = D;
};

vector_t PCA::transform(vector_t& x, int alpha){
	vector_t res = vector_t(alpha, 0);
	vector_t x_transformed = dot(this->eigenbasis, x);
	for (int i = 0; i < res.size(); ++i){
		res[i] = x_transformed[i];
	}
	return res;
}

VectorizedEntry PCA::transform(VectorizedEntry& x, int alpha){
	VectorizedEntry res = x;
	res.bag_of_words = this->transform(res.bag_of_words, alpha);
	return res;
};








