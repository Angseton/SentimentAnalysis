#include <gtest/gtest.h>
#include <vector>
#include "../src/PCA.h"

/* 
 * Test Linear Algebra operations/functions
 */


void EXPECT_NEAR_VECTOR(vector_t& x, vector_t& y, double epsilon){
    EXPECT_EQ(x.size(), y.size());
    for (int i = 0; i < x.size(); ++i){
        EXPECT_NEAR(x[i], y[i], epsilon);
    }
}

void EXPECT_NEAR_MATRIX(matrix& A, matrix& B, double epsilon){
    EXPECT_EQ(A.size(), B.size());
    EXPECT_EQ(A[0].size(), B[0].size());
    for (int i = 0; i < A.size(); ++i){
        for (int j = 0; j < A[0].size(); ++j){
            EXPECT_NEAR(A[i][j], B[i][j], epsilon);
        }
    }
}

void EXPECT_EIGENVECTOR(matrix& A, double lambda, vector_t& v, double epsilon){
    EXPECT_EQ(A.size(), v.size());
    vector_t Av = dot(A, v);
    vector_t lv = v;
    for (int i = 0; i < lv.size(); ++i){
        lv[i] *= lambda;
    }
    EXPECT_NEAR_VECTOR(Av, lv, epsilon);

}


 /* Test matrix vector multiplication */
 TEST(linalg_operations, test_dot_identity) {
     matrix A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
     vector_t x = {1, 0, 0};
     EXPECT_EQ(x, dot(A, x));
 }
 
 TEST(linalg_operations, test_dot_e1) {
     matrix A = {{1, 0, 0}, {1, 1, 0}, {2, 0, 1}};
     vector_t x = {1, 0, 0};
     vector_t res = {1, 1, 2};
     EXPECT_EQ(res, dot(A, x));
 }
 
 TEST(linalg_operations, test_dot_non_square) {
     matrix A = {{1, 0, 0}, {1, 1, 0}, {2, 0, 1}, {-1, 0, 1}};
     vector_t x = {1, 0, 0};
     vector_t res = {1, 1, 2, -1};
     EXPECT_EQ(res, dot(A, x));
 }
 
 /* Test vector vector multiplication */
 TEST(linalg_operations, test_transpose_dot_e1) {
     vector_t x = {1, 0, 0};
     EXPECT_EQ(1,  transpose_dot(x, x));
 }
 
 TEST(linalg_operations, test_transpose_dot_zero) {
     vector_t x = {0, 0, 0};
     EXPECT_EQ(0,  transpose_dot(x, x));
 }
 
 TEST(linalg_operations, test_transpose_dot_unit_vector) {
     vector_t x = {0.3, 0.3, 0.4};
     vector_t y = {1, 1, 1};
     EXPECT_EQ(1,  transpose_dot(x, y));
 }
 
 /* matrix transpose */
 TEST(linalg_operations, test_transpose_identity) {
     matrix A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
     EXPECT_EQ(A, transpose(A));
 }
 
 TEST(linalg_operations, test_transpose_2) {
     matrix A = {{0, 0, 1}, {0, 2, 0}, {3, 0, 0}};
     matrix res = {{0, 0, 3}, {0, 2, 0}, {1, 0, 0}};
     EXPECT_EQ(res, transpose(A));
 }
 
 TEST(linalg_operations, test_transpose_3) {
     matrix A = {{0, 0, 1}, {0, 2, 0}, {3, 0, 0}};
     matrix res = {{0, 0, 3}, {0, 2, 0}, {1, 0, 0}};
     EXPECT_EQ(res, transpose(A));
 }
 
 /* Vector 2-norm */
 TEST(linalg_operations, test_norm_zero) {
     vector_t x = {0, 0, 0};
     EXPECT_EQ(0, norm(x));
 }
 
 TEST(linalg_operations, test_norm_canonics) {
     vector_t e1 = {1, 0, 0};
     EXPECT_EQ(1, norm(e1));
     vector_t e2 = {1, 0, 0};
     EXPECT_EQ(1, norm(e2));
     vector_t e3 = {1, 0, 0};
     EXPECT_EQ(1, norm(e3));
 }
 
 TEST(linalg_operations, test_norm_unit_vector) {
     vector_t x = {0.5, 0.5, 0.5, 0.5};
     EXPECT_EQ(1, norm(x));
 }
 
 TEST(linalg_operations, test_normalize_canonics) {
     vector_t e1 = {1, 0, 0};
     vector_t e1_res = {1, 0, 0};
     normalize(e1);
     EXPECT_EQ(e1_res, e1);
     vector_t e2 = {1, 0, 0};
     vector_t e2_res = {1, 0, 0};
     normalize(e2);
     EXPECT_EQ(e2_res, e2);
     vector_t e3 = {1, 0, 0};
     vector_t e3_res = {1, 0, 0};
     normalize(e3);
     EXPECT_EQ(e3_res, e3);
 }
 
 TEST(linalg_operations, test_normalize_ones) {
     vector_t x = {1, 1, 1, 1};
     vector_t x_res = {0.5, 0.5, 0.5, 0.5};
     normalize(x);
     EXPECT_EQ(x_res, x);
 }
 
 TEST(pca, sample_mean_zeros) {
     matrix samples = {
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {0, 0, 0, 0}
     };
     vector_t res = {0, 0, 0, 0};
     EXPECT_EQ(res, sample_mean(samples));
 }
 
 TEST(pca, sample_mean_square_1) {
     matrix samples = {
         {1, 0, 1, 0},
         {1, 0, 0, 2},
         {1, 0, 3, 3},
         {1, 0, 0, 4}
     };
     vector_t res = {1, 0, 1, 2.25};
     EXPECT_EQ(res, sample_mean(samples));
 }
 
TEST(pca, sample_mean_square_2) {
     matrix samples = {
         {1, 4, 0.5, 3.5},
         {2, 3, 1.5, 2.5},
         {3, 2, 2.5, 1.5},
         {4, 1, 3.5, 0.5}
     };
     vector_t res = {2.5, 2.5, 2, 2};
     EXPECT_EQ(res, sample_mean(samples));
 }
 
TEST(pca, sample_mean_2) {
     matrix samples = {
         {0, 1, 2, 3, 4, 5, 6, 7, 8},
         {0, 1, 2, 3, 4, 5, 6, 7, 8},
         {0, 1, 2, 3, 4, 5, 6, 7, 8},
         {0, 1, 2, 3, 4, 5, 6, 7, 8}
     };
     vector_t res = {0, 1, 2, 3, 4, 5, 6, 7, 8};
     EXPECT_EQ(res, sample_mean(samples));
 }
 
TEST(pca, covariance_matrix_zeros) {
     matrix samples = {
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {0, 0, 0, 0},
         {0, 0, 0, 0}
     };
     EXPECT_EQ(samples, covariance_matrix(samples));
 }
 
TEST(pca, covariance_matrix_identity) {
     matrix samples = {
         {1, 0, 0, 0, 0},
         {0, 1, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {0, 0, 0, 1, 0},
         {0, 0, 0, 0, 1}
     };
 
     matrix res = {
         {0.2, -0.05, -0.05, -0.05, -0.05},
         {-0.05, 0.2, -0.05, -0.05, -0.05},
         {-0.05, -0.05, 0.2, -0.05, -0.05},
         {-0.05, -0.05, -0.05, 0.2, -0.05},
         {-0.05, -0.05, -0.05, -0.05, 0.2}
     };
     EXPECT_EQ(covariance_matrix(samples), covariance_matrix(samples));
 }
 
TEST(pca, dominant_eigenvalue_diagonal) {
     matrix A = {
         {5, 0, 0, 0, 0},
         {0, 1, 0, 0, 0},
         {0, 0, 1, 0, 0},
         {0, 0, 0, 1, 0},
         {0, 0, 0, 0, 1}
     };
     vector_t guess = {1, 1, 1, 1, 1};
     pair<double, vector_t> out = dominant_eigenvalue(A, guess, 20);
     EXPECT_NEAR(5, out.first, 0.001);
     EXPECT_EIGENVECTOR(A, out.first, out.second, 0.01);
 }
  
TEST(pca, dominant_eigenvalue_all_ones) {
     matrix A = {
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
     };
     vector_t guess = {1, 1, 1};
     pair<double, vector_t> out = dominant_eigenvalue(A, guess, 20);
     EXPECT_NEAR(3, out.first, 0.001);
     EXPECT_EIGENVECTOR(A, out.first, out.second, 0.01);
 }
 
 TEST(pca, dominant_eigenvalue_simmetric) {
     matrix A = {
         {1, 2, 3},
         {2, 2, 2},
         {3, 2, 3},
     };
     vector_t guess = {1, 1, 1};
     pair<double, vector_t> out = dominant_eigenvalue(A, guess, 20);
     EXPECT_NEAR(6.79624015, out.first, 0.00000001);
     EXPECT_EIGENVECTOR(A, out.first, out.second, 0.01);
 }

 TEST(pca, dominant_eigenvalue_matrix) {
     matrix A = {
         {3, -1, -0.5},
         {-1, 4/3, -1/3},
         {-0.5, -1/3, 1/3},
     };
     vector_t guess = {1, 1, 1};
     pair<double, vector_t> out = dominant_eigenvalue(A, guess, 20);
     EXPECT_NEAR(3.47, out.first, 0.01);
     EXPECT_EIGENVECTOR(A, out.first, out.second, 0.01);
 }
 
 TEST(pca, deflate_zeros) {
     matrix A = {
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
     };
     double lambda = 1;
     vector_t v = {1, 1, 1};
     deflate(A, v, lambda);
     matrix res = {
         {-1, -1, -1},
         {-1, -1, -1},
         {-1, -1, -1},
     };
     EXPECT_EQ(A, res);
 }
 
 TEST(pca, deflate_matrix) {
     matrix A = {
         {1, 2, 3},
         {4, 5, 6},
         {7, 8, 9},
     };
     double lambda = 2;
     vector_t v = {1, 2, 3};
     deflate(A, v, lambda);
     matrix res = {
         {-1, -2, -3},
         { 0, -3, -6},
         { 1, -4, -9},
     };
     EXPECT_EQ(A, res);
 }


TEST(pca, power_method) {
    matrix A = {
        {3, -1, -0.5},
        {-1, 4/3, -1/3},
        {-0.5, -1/3, 1/3},
    };
    matrix A_copy = A;
    vector_t guess = {1, 1, 1};
    pair<double, vector_t> out;

    out = dominant_eigenvalue(A, guess, 20);
    EXPECT_EIGENVECTOR(A_copy, out.first, out.second, 0.0001);
    
    deflate(A, out.second, out.first);
    out = dominant_eigenvalue(A, guess, 20);
    EXPECT_EIGENVECTOR(A_copy, out.first, out.second, 0.0001);
    
    deflate(A, out.second, out.first);
    out = dominant_eigenvalue(A, guess, 20);
    EXPECT_EIGENVECTOR(A_copy, out.first, out.second, 0.0001);
}