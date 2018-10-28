#include <cmath>
#include <vector>
#include <random>
#include "linalg.h"
#include "../src_catedra/types.h"

#define DELTA_MAX 10e-4
#define ITERS_MAX 1000

class PCA {
public:
	PCA();
	void fit(matrix& X);
	void fit(VectorizedEntriesMap& dataset);
	vector_t transform(vector_t& x, int alpha);
	VectorizedEntry transform(VectorizedEntry& x, int alpha);
	void transform(
		VectorizedEntriesMap& dataset, 
		VectorizedEntriesMap& transformed_dataset,
		int alpha
	);
	vector_t get_eigenvalues();
	matrix get_eigenbasis();

private:
	matrix eigenbasis;
	vector_t eigenvalues;
	
};

PCA::PCA(){}

vector_t PCA::get_eigenvalues(){
	return this->eigenvalues;
}

matrix PCA::get_eigenbasis(){
	return this->eigenbasis;
}

void PCA::fit(matrix& X){
	// Build samples covariance matrix
	std::cerr << "Building covariance matrix...           " << '\r';
	matrix M = covariance_matrix(X);
	// Eigenvector and eigenvalues matrices
	this->eigenbasis = matrix(M.size(), vector_t(M.size(), 0)); // Eigenvector matrix.
	this->eigenvalues = vector_t(M.size(), 0); // Eigenvalues.
	std::cerr << "Computing eigenvector basis             " << '\r';
	for (int i = 0; i < M.size(); ++i){
		std::cerr << "Computing eigenvector: " << i << '/' << M.size() <<"              \r";
		/*
		 * Compute dominant eigenvalue and its correspondent eigenvector
		 * then deflate the covariance matrix.
		 */
		vector_t v_i = generate_random_guess(M.size());
		pair<double, vector_t> component = dominant_eigenvalue(M, v_i, ITERS_MAX, DELTA_MAX);
		this->eigenvalues[i] = component.first;
		this->eigenbasis[i] = component.second;
		deflate(M, component.second, component.first);
	}

	std::cerr << "                                                " <<'\r';
};

void PCA::fit(VectorizedEntriesMap& dataset){
	matrix X = matrix(
		dataset.size(),
		vector_t(dataset.begin()->second.bag_of_words.size(), 0));
	int i = 0;
	for (auto sample = dataset.begin(); sample != dataset.end(); ++sample){
		X[i] = sample->second.bag_of_words;
		i++;
	}
	this->fit(X);
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
}

void PCA::transform(
	VectorizedEntriesMap& dataset,
	VectorizedEntriesMap& transformed_dataset,
	int alpha){
	for (auto it = dataset.begin(); it != dataset.end(); ++it){
		transformed_dataset[it->first] = this->transform(it->second, alpha);
	}
}
