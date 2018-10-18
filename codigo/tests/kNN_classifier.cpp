#include <gtest/gtest.h>
#include <string>
#include "../vector_builder.h"
#include "../KNNClassifier.h"

TEST(kNN_classifier, test_k_1) {
    VectorizedEntry entry_1;
    entry_1.bag_of_words = {1, 1, 1};
    entry_1.is_positive = true;

    VectorizedEntry entry_2;
    entry_2.bag_of_words = {0, 1, 1};
    entry_2.is_positive = false;

    VectorizedEntry entry_3;
    entry_3.bag_of_words = {3, 0, 0};
    entry_3.is_positive = false;

    VectorizedEntry entry_4;
    entry_4.bag_of_words = {2, 0, 2};
    entry_4.is_positive = false;

    VectorizedEntry query;
    query.bag_of_words = {1, 1, 0};

    VectorizedEntriesMap train_map = {{1,entry_1},{2,entry_2},{3,entry_3},{4,entry_4}};

    KNNClassifier knnTester(train_map);

    EXPECT_TRUE(knnTester.predict(query, 1));
}

TEST(kNN_classifier, test_k_3) {
    VectorizedEntry entry_1;
    entry_1.bag_of_words = {1, 1, 1};
    entry_1.is_positive = true;

    VectorizedEntry entry_2;
    entry_2.bag_of_words = {0, 1, 1};
    entry_2.is_positive = false;

    VectorizedEntry entry_3;
    entry_3.bag_of_words = {3, 0, 0};
    entry_3.is_positive = false;

    VectorizedEntry entry_4;
    entry_4.bag_of_words = {2, 0, 2};
    entry_4.is_positive = false;

    VectorizedEntry query;
    query.bag_of_words = {1, 1, 0};

    VectorizedEntriesMap train_map = {{1,entry_1},{2,entry_2},{3,entry_3},{4,entry_4}};

    KNNClassifier knnTester(train_map);

    EXPECT_FALSE(knnTester.predict(query, 3));
}

TEST(kNN_classifier, test_empate) { 
    //Para testear la política de empate. 
    // Actualmente si hay empate, el resultado es negativo.
    VectorizedEntry entry_1;
    entry_1.bag_of_words = {1, 1, 1};
    entry_1.is_positive = true;

    VectorizedEntry entry_2;
    entry_2.bag_of_words = {0, 1, 1};
    entry_2.is_positive = false;

    VectorizedEntry entry_3;
    entry_3.bag_of_words = {3, 0, 0};
    entry_3.is_positive = true;

    VectorizedEntry entry_4;
    entry_4.bag_of_words = {2, 0, 2};
    entry_4.is_positive = false;

    VectorizedEntry query;
    query.bag_of_words = {1, 1, 0};

    VectorizedEntriesMap train_map = {{1,entry_1},{2,entry_2},{3,entry_3},{4,entry_4}};

    KNNClassifier knnTester(train_map);

    EXPECT_FALSE(knnTester.predict(query, 4));
}