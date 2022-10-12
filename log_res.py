from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing


def get_vocabulary(dataset: pd.DataFrame):
    wcv = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    ccv = CountVectorizer(analyzer='char', ngram_range=(2, 5))
    vocab = FeatureUnion([('word_ngram_counts', wcv), ('char_ngram_counts', ccv)])
    vocab.fit([t for t in dataset["text"]])
    return vocab


class LogResClassifier(object):

    def __init__(self, preprocessing=[], vocabulary=None, *, max_iter=1000):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        self.lrc = LogisticRegression(max_iter=max_iter)

    def train(self, trainingset: pd.DataFrame, kfold=0, verbose=False):
        if self.vocab is None:
            self.vocab = get_vocabulary(trainingset)

        if kfold > 0:
            gold = []
            pred = []
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(trainingset):
                trainingset = trainingset.reset_index()  # TODO make the train-test split work
                X_train = self.__create_features([text for text in trainingset[train_index.__contains__(trainingset["index"])]["text"]])
                X_test = self.__create_features([text for text in trainingset[test_index.__contains__(trainingset["index"])]["text"]])
                y_train = [s for s in trainingset[train_index.__contains__(trainingset["index"])]["score"]]
                y_test = [s for s in trainingset[test_index.__contains__(trainingset["index"])]["score"]]

                # Train the model
                self.lrc.fit(X_train, y_train)  # Training the model

                # Test model
                predict = self.lrc.predict(X_test)
                gold += y_test
                pred += list(predict)
                if verbose:
                    print(f"Fold no. {i}")
                    print(f"Accuracy = {accuracy_score(y_test, predict)}")
                    print(f"Kappa = {cohen_kappa_score(y_test, predict, weights='quadratic')}")
                i += 1
            return gold, pred
        else:
            count_matrix = self.__create_features([t for t in trainingset["text"]])
            y = [s for s in trainingset["score"]]
            self.lrc.fit(count_matrix, y)
            return [], []

    def __create_features(self, data: List[str]):

        count_matrix = self.vocab.transform( # TODO AH: check whether this is really necessary or whether you can use sklearn transform
            [preprocessing.compose(*self.preprocessing)(text) for text in data])
        return count_matrix

    def predict(self, text: str) -> int:
        return self.lrc.predict(self.__create_features([text]))[0]

