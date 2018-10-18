#ifndef GASTICODES_KNNCLASSIFIER_H
#define GASTICODES_KNNCLASSIFIER_H

#include "vector_builder.h"
#include <math.h>
#include <algorithm>

using namespace std;

class KNNClassifier{

public:
    explicit KNNClassifier(VectorizedEntriesMap& train_entries);
    bool predict(VectorizedEntry& x, int k);

private:
    VectorizedEntriesMap _train_entries;
    VectorizedEntry vector_substraction(VectorizedEntry x, VectorizedEntry b);
    pair<double, bool> norm(VectorizedEntry a);
    bool cmp(pair<double, bool> a, pair<double, bool> b);
    bool mode(vector<pair<double, bool>> norms, int k);

};

KNNClassifier::KNNClassifier(VectorizedEntriesMap& train_entries) : _train_entries(train_entries){}

bool KNNClassifier::predict(VectorizedEntry &x, int k) {
    //Armo un vector de normas con sus flags
    vector<pair<double, bool>> norms;
    for(const auto& pair : this->_train_entries){
        norms.push_back(norm(vector_substraction(x, pair.second)));
    }

    //Ordeno el vector
    sort(norms.begin(), norms.end(),[this] (pair<double, bool> a, pair<double, bool> b) {return cmp(a, b);});

    //Busco y devuelvo la moda para los k primeros elementos
    return this->mode(norms, k);

}

bool KNNClassifier::mode(vector<pair<double, bool>> norms, int k){
    unsigned int positiveCount = 0;
    for(unsigned int i = 0; i < k; i++){
        if(norms[i].second) positiveCount++;
    }
    unsigned int negativeCount = k - positiveCount;
    return positiveCount > negativeCount;
}

bool KNNClassifier::cmp(pair<double, bool> a, pair<double, bool> b) {
    return (a.first < b.first);
}

VectorizedEntry KNNClassifier::vector_substraction(VectorizedEntry x, VectorizedEntry b) {
    VectorizedEntry ans;
    vector<double> ans_bag_of_words;
    ans.bag_of_words = ans_bag_of_words;
    for(unsigned int i = 0; i < x.bag_of_words.size(); i++){
        ans.bag_of_words.push_back(x.bag_of_words[i] - b.bag_of_words[i]);
    }
    ans.is_positive = b.is_positive;
    return ans;
}

pair<double, bool> KNNClassifier::norm(VectorizedEntry a) {
    double squareSum = 0;
    for(auto& a_i : a.bag_of_words){
        squareSum += pow(a_i, 2.0);
    }

    return make_pair(sqrt(squareSum), a.is_positive);
}

#endif //GASTICODES_KNNCLASSIFIER_H
