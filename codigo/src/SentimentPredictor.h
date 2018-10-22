#ifndef SENTIMENTPREDICTOR_H
#define SENTIMENTPREDICTOR_H

#include "PCA.h"

class SentimentPredictor{
public:

    SentimentPredictor();
    void predictDataSet(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha, string output_file);

private:

    void applyKNN(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
                    string output_file);
    void printData(int tp, int fp, int tn, int fn, int amount, int method, int k, int alpha);
};

SentimentPredictor::SentimentPredictor() {}

void SentimentPredictor::predictDataSet(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
                                        string output_file) {
    if(method == 0){
        //kNN
        applyKNN(trainSet, testSet, method, k, alpha, output_file);
    } else if (method == 1) {
        //PCA + kNN
        PCA pca;
        pca.fit(trainSet);
        VectorizedEntriesMap transformedTrainSet;
        for(auto& pair : trainSet){
            transformedTrainSet[pair.first] = pca.transform(pair.second, alpha);
        }
        applyKNN(transformedTrainSet, testSet, method, k, alpha, output_file);
    } else {
        throw std::domain_error("No such method.");
    }
}

/** Helper Methods **/

void SentimentPredictor::applyKNN(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
                                  string output_file) {
    int tp = 0, fp = 0, tn = 0, fn = 0, amount = 0;
    KNNClassifier knn = KNNClassifier(trainSet);
    ofstream ofs;
    ofs.open(output_file);
    ofs << "Test entry" << " Prediction" << endl;
    for (auto &pair : testSet) {
        bool prediction = knn.predict(pair.second, k);
        bool realSentiment = pair.second.is_positive;
        if (realSentiment && prediction) tp++;
        else if (realSentiment && !prediction) fn++;
        else if (!realSentiment && prediction) fp++;
        else if (!realSentiment && !prediction) tn++;
        amount++;
        ofs << pair.first << " " << prediction << endl;
    }
    printData(tp, fp, tn, fn, amount, method, k, alpha);
}

void SentimentPredictor::printData(int tp, int fp, int tn, int fn, int amount, int method, int k, int alpha) {
    if(method == 1){
        //kNN + PCA, we output alpha
        cout << "El método utilizado fue: kNN + PCA" << endl;
        cout << "Alpha: " << alpha << endl;
    } else {
        cout << "El método utilizado fue: kNN" << endl;
        cout << "Alpha: --" << endl;
    }
    cout << "k: " << k << endl;
    cout << "Tamaño del test: " << amount << endl;
    cout << "tp: " << tp << endl;
    cout << "fp: " << fp << endl;
    cout << "tn: " << tn << endl;
    cout << "fn: " << fn << endl;
    cout << "Recall: " << (tp / (double)(tp + fp)) << endl;
    cout << "Precision: " << (tp / (double)(tp + fn)) << endl;
}

#endif
