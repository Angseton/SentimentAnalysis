#ifndef GASTICODES_KNNCLASSIFIER_H
#define GASTICODES_KNNCLASSIFIER_H

#include "../src_catedra/vector_builder.h"
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
    pair<double, bool> distance(const VectorizedEntry& x, const VectorizedEntry& sample);

};

KNNClassifier::KNNClassifier(VectorizedEntriesMap& train_entries) : _train_entries(train_entries){}

bool KNNClassifier::predict(VectorizedEntry &x, int k) {
    //Armo un vector de normas con sus flags
    vector<pair<double, bool>> norms;
    for(const auto& pair : this->_train_entries){
        norms.push_back(distance(pair.second, x));
    }
    //Ordeno el vector
    sort(norms.begin(), norms.end(),[this] (pair<double, bool> a, pair<double, bool> b) {return cmp(a, b);});

    //Busco y devuelvo la moda para los k primeros elementos
    return this->mode(norms, k);

}

bool KNNClassifier::mode(vector<pair<double, bool>>& norms, int k){
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


pair<double, bool> KNNClassifier::distance(const VectorizedEntry& x, const VectorizedEntry& sample) {
    double squareSum = 0;
    for(unsigned int i = 0; i < x.bag_of_words.size(); i++){
        squareSum += pow(x.bag_of_words[i] - sample.bag_of_words[i], 2.0);
    }
    return make_pair(sqrt(squareSum), x.is_positive);
}

#endif //GASTICODES_KNNCLASSIFIER_H
