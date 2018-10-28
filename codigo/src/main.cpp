#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "KNNClassifier.h"
#include "PCA.h"

string getCmdOption(char ** begin, char ** end, const std::string & option){
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end){
        string cmd(*itr);
        return cmd;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option){
    return std::find(begin, end, option) != end;
}

int main(int argc, char * argv[]){

    string dataset_file;
    int mode = 0;
    int k = 5;
    int alpha = 30;
    double threshold_frecuency_low = 0.01;
    double threshold_frecuency_high = 0.99;
    string output_file;
    string basis_file;

    if(cmdOptionExists(argv, argv+argc, "-i")){
        dataset_file = getCmdOption(argv, argv + argc, "-i");
    }
    if(cmdOptionExists(argv, argv+argc, "-o")){
        output_file = getCmdOption(argv, argv + argc, "-o");
    }
    if(cmdOptionExists(argv, argv+argc, "-m")){
        mode = stoi(getCmdOption(argv, argv + argc, "-m"));
    }
    if(cmdOptionExists(argv, argv+argc, "-k")){
        k = stoi(getCmdOption(argv, argv + argc, "-k"));
    }
    if(cmdOptionExists(argv, argv+argc, "-a")){
        alpha = stoi(getCmdOption(argv, argv + argc, "-a"));
    }
    if(cmdOptionExists(argv, argv+argc, "-f_low")){
        threshold_frecuency_low = stod(getCmdOption(argv, argv + argc, "-f_low"));
    }
    if(cmdOptionExists(argv, argv+argc, "-f_high")){
        threshold_frecuency_high = stod(getCmdOption(argv, argv + argc, "-f_high"));
    }

    cout <<  "Dataset File: " << dataset_file << endl;
    cout <<  "mode: " << mode << endl;
    cout <<  "k: " << k << endl;
    cout <<  "alpha: " << alpha << endl;
    cout <<  "threshold_frecuency_low: " << threshold_frecuency_low << endl;
    cout <<  "threshold_frecuency_high: " << threshold_frecuency_high << endl;

    auto filter_out = [threshold_frecuency_low, threshold_frecuency_high] 
    (const int token, const FrecuencyVocabularyMap & vocabulary) {
        /**
         *  Lambda para usar como filtro de vocabulario
         *  Retorna `true` si `token` debe eliminarse
         *  Retorna `false` si `token` no debe eliminarse
         **/
        double token_frecuency = vocabulary.at(token);
        return token_frecuency < threshold_frecuency_low || 
               token_frecuency > threshold_frecuency_high;
    };
    
    VectorizedEntriesMap train_entries;
    VectorizedEntriesMap test_entries;
    VectorizedEntriesMap transformed_train_entries;
    VectorizedEntriesMap transformed_test_entries;
    
    build_vectorized_datasets(dataset_file, train_entries, test_entries, filter_out);
    
    auto start = chrono::system_clock::now();

    switch(mode){
        case 0: // KNN no pca
            transformed_train_entries = train_entries;
            transformed_test_entries = test_entries;
            break;
        case 1: // KNN with pca
            PCA pca = PCA();
            pca.fit(train_entries);
            pca.transform(train_entries, transformed_train_entries, alpha);
            pca.transform(test_entries, transformed_test_entries, alpha);
            break;
    }
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    int amount = 0;

    KNNClassifier clf = KNNClassifier(transformed_train_entries);
    for (auto it = transformed_test_entries.begin(); it != transformed_test_entries.end(); it++) {
        std::cerr << "Prediciendo " << amount << " / " << transformed_test_entries.size() << '\r';
        bool label = it->second.is_positive;
        bool predi = clf.predict(it->second, 5);
        if (label && predi) tp++;
        else if (label && !predi) fn++;
        else if (!label && predi) fp++;
        else if (!label && !predi) tn++;
        amount++;
    }

    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end-start;

    std::cout
        << "                                    " << std::endl
        << "tp: " << tp << std::endl
        << "fp: " << fp << std::endl
        << "tn: " << tn << std::endl
        << "fn: " << fn << std::endl
        << "Accuracy: " << ((tp + tn)/ (double)(tp + fp + tn + fn)) << std::endl
        << "Precision: " << (tp / (double)(tp + fn)) << std::endl
        << "Recall: " << (tp / (double)(tp + fp)) << std::endl
        << "F1: " << ((2 * tp) / (double)(2 * tp + fp + fn)) << std::endl
        << "Time: "<< elapsed_seconds.count() << endl;
    return 0;
}