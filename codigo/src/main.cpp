#include "../src_catedra/vector_builder.h"
#include "KNNClassifier.h"
#include "PCA.h"


int main() {
    auto filter_out = [] (const int token, const FrecuencyVocabularyMap & vocabulary) {
        /**
         *  Lambda para usar como filtro de vocabulario
         *  Retorna `true` si `token` debe eliminarse
         *  Retorna `false` si `token` no debe eliminarse
         **/
        double token_frecuency = vocabulary.at(token);
        return token_frecuency < 0.01 || token_frecuency > 0.99;
    };
    VectorizedEntriesMap train_entries;
    VectorizedEntriesMap test_entries;
    build_vectorized_datasets(train_entries, test_entries, filter_out);
    int N = train_entries.begin()->second.bag_of_words.size();

    std::cerr
        << "N: " << N << std::endl
        << "Dataset de entrenamiento: " << train_entries.size() << " entradas" << std::endl
        << "Dataset de testeo: " << test_entries.size() << " entradas" << std::endl;

    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    int amount = 0;
    KNNClassifier clf = KNNClassifier(train_entries);
    for (auto it = test_entries.begin(); it != test_entries.end(); it++) {
        std::cerr << "Prediciendo " << amount << " / " << train_entries.size() << '\r';
        bool label = it->second.is_positive;
        bool predi = clf.predict(it->second, 5);
        if (label && predi) tp++;
        else if (label && !predi) fn++;
        else if (!label && predi) fp++;
        else if (!label && !predi) tn++;
        amount++;
    }

    std::cerr
        << "                                    " << std::endl
        << tp << std::endl
        << fp << std::endl
        << tn << std::endl
        << fn << std::endl
        << "Recall: " << (tp / (double)(tp + fp)) << std::endl
        << "Precision: " << (tp / (double)(tp + fn)) << std::endl;

    return 0;
}

