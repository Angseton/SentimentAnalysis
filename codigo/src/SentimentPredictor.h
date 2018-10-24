#ifndef SENTIMENTPREDICTOR_H
#define SENTIMENTPREDICTOR_H

#include "PCA.h"
#include <chrono>

class SentimentPredictor{
public:

    SentimentPredictor();
    void predictDataSet(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha, string output_file, bool log);

private:
    struct dataInfo{
        int fp, fn, tp, tn, amount;
    };
    dataInfo applyKNN(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
                    string output_file);
    void printData(int tp, int fp, int tn, int fn, int amount, int method, int k, int alpha);
};

SentimentPredictor::SentimentPredictor() {}

void SentimentPredictor::predictDataSet(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
                                        string output_file, bool log) {

    auto start = std::chrono::high_resolution_clock::now();
    ofstream log_data;
    dataInfo data;
    if(method == 0){
        //kNN
        data = applyKNN(trainSet, testSet, method, k, alpha, output_file);
    } else if (method == 1) {
        //PCA + kNN
        PCA pca;
        pca.fit(trainSet);
        VectorizedEntriesMap transformedTrainSet;
        for(auto& pair : trainSet){
            transformedTrainSet[pair.first] = pca.transform(pair.second, alpha);
        }
        VectorizedEntriesMap transformedTestSet;
        for(auto& pair : testSet){
            transformedTestSet[pair.first] = pca.transform(pair.second, alpha);
        }
        data = applyKNN(transformedTrainSet, transformedTestSet, method, k, alpha, output_file);
    } else {
        throw std::domain_error("No such method.");
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    double elapsedTime = elapsed.count();

    //If log mode on:
    if(log){
        ofstream ofs;
        ofs.open("log.txt");
        if(method == 1){
            //kNN + PCA, we output alpha
            ofs << "El método utilizado fue: kNN + PCA" << endl;
            ofs << "Alpha: " << alpha << endl;
        } else {
            ofs << "El método utilizado fue: kNN" << endl;
            ofs << "Alpha: --" << endl;
        }
        ofs << "k: " << k << endl;
        ofs << "Tamaño del test: " << data.amount << endl;
        ofs << "tp: " << data.tp << endl;
        ofs << "fp: " << data.fp << endl;
        ofs << "tn: " << data.tn << endl;
        ofs << "fn: " << data.fn << endl;
        ofs << "Recall: " << (data.tp / (double)(data.tp + data.fp)) << endl;
        ofs << "Precision: " << (data.tp / (double)(data.tp + data.fn)) << endl;
        ofs << "Elapsed time: " << elapsedTime << endl;
        ofs.close();
    }
}

/** Helper Methods **/

typename SentimentPredictor::dataInfo SentimentPredictor::applyKNN(VectorizedEntriesMap& trainSet, VectorizedEntriesMap& testSet, int method, int k, int alpha,
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
    ofs.close();
    printData(tp, fp, tn, fn, amount, method, k, alpha);

    dataInfo data;
    data.tp = tp;
    data.fp = fp;
    data.tn = tn;
    data.fn = fn;
    data.amount = amount;
    return data;
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
