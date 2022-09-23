
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing
from data import *


def get_vocabulary(*datasets, lang="en"):
    langgraber = lambda x: x.og_text
    if lang == "en":
        langgraber = lambda x: x.en_text
    if lang == "de":
        langgraber = lambda x: x.de_text
    if lang == "es":
        langgraber = lambda x: x.es_text
    set = []
    for d in datasets:
        set.extend(d.copy())

    wcv = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    ccv = CountVectorizer(analyzer='char', ngram_range=(2, 5))
    vocab = FeatureUnion([('word_ngram_counts', wcv), ('char_ngram_counts', ccv)])
    vocab.fit([langgraber(data) for data in set])
    return vocab


class LogResClassifier(object):

    def __init__(self, preprocessing=[], lang="en", vocabulary=None, *, max_iter=1000):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        self.svc = LogisticRegression(max_iter=max_iter)
        self.lang = lang

    def train(self, trainingset, kfold=False, verbose=False):
        if self.vocab is None:
            self.vocab = get_vocabulary(trainingset, lang=self.lang)

        if kfold:
            gold = []
            pred = []
            kf = KFold(n_splits=10, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(trainingset):
                X_train = self.__create_features([trainingset[i] for i in train_index])
                X_test = self.__create_features([trainingset[i] for i in test_index])
                y_train = [trainingset[i].gold_score for i in train_index]
                y_test = [trainingset[i].gold_score for i in test_index]

                # Train the model
                self.svc.fit(X_train, y_train)  # Training the model

                # Test model
                predict = self.svc.predict(X_test)
                gold += y_test
                pred += list(predict)
                if verbose:
                    print(f"Fold no. {i}")
                    print(f"Accuracy = {accuracy_score(y_test, predict)}")
                    print(f"Kappa = {cohen_kappa_score(y_test, predict, weights='quadratic')}")
                i += 1
            return gold, pred
        else:
            count_matrix = self.__create_features(trainingset)
            y = [data_entry.gold_score for data_entry in trainingset]
            self.svc.fit(count_matrix, y)
            return [], []

    def __create_features(self, data: List[CrossLingualDataEntry]):
        langgraber = lambda x: (x.og_text, x.lang)
        if self.lang == "en":
            langgraber = lambda x: (x.en_text, "en")
        if self.lang == "de":
            langgraber = lambda x: (x.de_text, "de")
        if self.lang == "es":
            langgraber = lambda x: (x.es_text, "es")

        count_matrix = self.vocab.transform(
            [preprocessing.compose(*self.preprocessing)(*langgraber(data_entry)) for data_entry in data])
        return count_matrix

    def predict(self, data: CrossLingualDataEntry) -> int:
        return self.svc.predict(self.__create_features([data]))[0]

