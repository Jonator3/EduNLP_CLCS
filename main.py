import os.path
import sys
from typing import List

import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate, KFold
import pickle
import csv

import preprocessing


class CrossLingualDataEntry(object):

    def __init__(self, id, lang, set, gold_score, og_text, en_text):
        self.id = id
        self.lang = lang
        self.set = int(set)
        self.gold_score = int(gold_score)
        self.og_text = og_text
        self.en_text = en_text

    def __str__(self):
        return f"CLDE: {self.id} in set {self.set}: {self.gold_score}"


class CrossLingualContendScoring(object):

    def __init__(self, preprocessing=[], vocabulary=None):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        self.svc = svm.SVC()


    def train(self, trainingset, kfold=False, verbose=False):
        if self.vocab is None:
            self.vocab = get_vocabulary(trainingset)
        if kfold:
            kf = KFold(n_splits=10, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(trainingset):
                X_train = self.__create_features([trainingset[i] for i in train_index])
                X_test = self.__create_features([trainingset[i] for i in test_index])
                y_train = [trainingset[i].gold_score for i in train_index]
                y_test = [trainingset[i].gold_score for i in test_index]

                # Train the model
                self.svc.fit(X_train, y_train)  # Training the model
                if verbose:
                    print(f"Fold no. {i}")
                    print(f"Accuracy = {accuracy_score(y_test, self.svc.predict(X_test))}")
                    print(f"Kappa = {cohen_kappa_score(y_test, self.svc.predict(X_test), weights='quadratic')}")
                i += 1
        else:
            count_matrix = self.__create_features(trainingset)
            y = [data_entry.gold_score for data_entry in trainingset]
            self.svc.fit(count_matrix, y)

    def __create_features(self, data: List[CrossLingualDataEntry]):
        count_matrix = self.vocab.transform(
            [preprocessing.compose(*self.preprocessing)(data_entry.en_text) for data_entry in data])
        return count_matrix

    def predict(self, data: CrossLingualDataEntry) -> int:
        return self.svc.predict(self.__create_features([data]))[0]


class Combined_CLCS(object):

    def __init__(self, clcs_list: List[CrossLingualContendScoring], indicator: callable):
        self.clcs_list = clcs_list.copy()
        self.indicator = indicator

    def predict(self, data: CrossLingualDataEntry) -> int:
        return self.clcs_list[self.indicator(data)].predict(data)


def load_data(input_path: str) -> List[CrossLingualDataEntry]:
    data = []
    reader = csv.reader(open(input_path, "r"))
    reader.__next__()  # skip head
    for row in reader:
        entry = CrossLingualDataEntry(row[0], row[4], row[1], row[2], row[3], row[5])
        data.append(entry)
    return data


def validate(svm, dataset: List[CrossLingualDataEntry]):
    predict = []
    gold = []
    for data in dataset:
        p = svm.predict(data)
        g = data.gold_score
        predict.append(p)
        gold.append(g)
    return gold, predict


def get_vocabulary(*datasets):
    set = []
    for d in datasets:
        set.extend(d.copy())

    vocab = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    vocab.fit([data.en_text for data in set])
    return vocab


def print_validation(gold, predict):
    mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(len(gold)):
        g = gold[i]
        p = predict[i]
        mat[g][p] += 1
    print("Gold/\tPrediction")
    print("\t|", "0\t\t|", "1\t\t|", "2\t\t|", "3")
    for g, pre in enumerate(mat):
        print(g, "\t|", pre[0], "\t|", pre[1], "\t|", pre[2], "\t|", pre[3])
    print("--------------------------------------")
    print("quadratic_kappa =", round(cohen_kappa_score(gold, predict, weights="quadratic"), 3))
    print("accuracy =", round(accuracy_score(gold, predict), 3))
    print("")


def separate_set(dataset: List[CrossLingualDataEntry]):
    output = [[], [], [], [], [], [], [], [], [], []]
    for d in dataset:
        output[d.set - 1].append(d)
    return output


def main(ignore_en_only_prompt=False):
    en_train = separate_set(load_data("data/en_train.csv"))
    en_test = separate_set(load_data("data/en_test.csv"))
    de_test = separate_set(load_data("data/de.csv"))
    es_test = separate_set(load_data("data/es.csv"))
    de_train = de_test.copy()
    es_train = es_test.copy()

    for set in range(10):
        if ignore_en_only_prompt:
            if not [0, 1, 9].__contains__(set):
                continue
        print("Training set", set + 1)
        file_name = f"only_en_{str(set + 1)}.clcs"
        svc = None
        if os.path.isfile(file_name):
            print("File found")
            svc = pickle.load(open(file_name, "rb"))
            print("loading done")
        else:
            print("no File found")
            preproc = [preprocessing.lower]
            svc = CrossLingualContendScoring(preproc)
            svc.train(en_train[set])
            #svc.train(de_train[set] + es_train[set], kfold=True)
            pickle.dump(svc, open(file_name, "wb"))  # save the svm to file
            print("Training Done!")
            print("")

        gold, predict = validate(svc, en_test[set])
        print("English:")
        print_validation(gold, predict)

        gold, predict = validate(svc, es_test[set])
        print("Spanish:")
        print_validation(gold, predict)

        gold, predict = validate(svc, de_test[set])
        print("German:")
        print_validation(gold, predict)
        print("")


if __name__ == "__main__":
    main(
        ignore_en_only_prompt=True
    )
