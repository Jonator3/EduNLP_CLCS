import os.path
import sys
from typing import List

import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
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

    def __init__(self, trainingset, preprocessing=[], vocabulary=None):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        if vocabulary is None:
            self.vocab = get_vocabulary(trainingset)
        count_matrix = self.__create_features(trainingset)
        self.svc = svm.SVC()
        self.svc.fit(count_matrix, [data_entry.gold_score for data_entry in trainingset])

    def __create_features(self, data: List[CrossLingualDataEntry]):
        count_matrix = self.vocab.transform([preprocessing.compose(*self.preprocessing)(data_entry.en_text) for data_entry in data])
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
    mat = []
    for i1 in range(4):
        row = []
        for i2 in range(4):
            row.append(0)
        mat.append(row)
    for data in dataset:
        predict = svm.predict(data)
        gold = data.gold_score
        mat[gold][predict] += 1
    return mat


def get_vocabulary(*datasets):
    set = []
    for d in datasets:
        set.extend(d.copy())

    vocab = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    vocab.fit([data.en_text for data in set])
    return vocab


def print_validation(mat):
    print("Gold/\tPrediction")
    print("\t|", "0\t\t|", "1\t\t|", "2\t\t|", "3")
    for g, pre in enumerate(mat):
        print(g, "\t|", pre[0], "\t|", pre[1], "\t|", pre[2], "\t|", pre[3])
    print("kappa =", round(calc_kappa(mat), 3))
    print("acc =", round(calc_acc(mat), 3))
    print("")


def calc_acc(mat):
    n = len(mat)
    total = 0
    p0 = 0
    for i in range(n):
        p0 += mat[i][i]
        temp = sum(mat[i])
        total += temp
    if total == 0:
        return -1
    return p0 / total


def calc_kappa(mat):
    n = len(mat)
    total = 0
    p0 = 0
    pe = 0
    for i in range(n):
        p0 += mat[i][i]
        temp = sum(mat[i])
        total += temp
        pe += temp + sum([mat[m][i] for m in range(n)])
    if total == 0:
        return -1
    p0 = p0/total
    pe = pe/(total**2)
    kappa = (p0 - pe) / (1 - pe)
    return kappa


def separate_set(dataset: List[CrossLingualDataEntry]):
    output = [[], [], [], [], [], [], [], [], [], []]
    for d in dataset:
        output[d.set - 1].append(d)
    return output


def main_make_svc():
    en_train = separate_set(load_data("data/en_train.csv"))
    en_test = separate_set(load_data("data/en_test.csv"))
    de_test = separate_set(load_data("data/de.csv"))
    es_test = separate_set(load_data("data/es.csv"))
    de_train = [[], [], [], [], [], [], [], [], [], []]
    for set in range(10):
        de_train[set] = de_test[set][len(de_test[set]) // 10:]
        de_test[set] = de_test[set][:(len(de_test[set]) // 10) + 1]
    es_train = [[], [], [], [], [], [], [], [], [], []]
    for set in range(10):
        es_train[set] = es_test[set][len(es_test[set]) // 10:]
        es_test[set] = es_test[set][:(len(es_test[set]) // 10) + 1]

    for set in range(10):
        print("Training set", set + 1)
        file_name = f"all_lang_{str(set + 1)}.clcs"
        svc = None
        if os.path.isfile(file_name):
            print("File found")
            svc = pickle.load(open(file_name, "rb"))
            print("loading done")
        else:
            print("no File found")
            train = en_train[set] + de_train[set] + es_train[set]
            preproc = [preprocessing.lower]
            svc = CrossLingualContendScoring(train, preproc)
            pickle.dump(svc, open(file_name, "wb"))  # save the svm to file
            print("Training Done!")

        print("English:")
        en_val = validate(svc, en_test[set])
        print_validation(en_val)
        print("Spanish:")
        es_val = validate(svc, es_test[set])
        print_validation(es_val)
        print("German:")
        de_val = validate(svc, de_test[set])
        print_validation(de_val)
        print("")

def main_eval_total():
    en_train = separate_set(load_data("data/en_train.csv"))
    en_test = load_data("data/en_test.csv")
    de_data = separate_set(load_data("data/de.csv"))
    es_data = separate_set(load_data("data/es.csv"))
    de_test = []
    de_train = [[], [], [], [], [], [], [], [], [], []]
    for set in range(10):
        de_train[set] = de_data[set][len(de_data[set]) // 10:]
        de_test.extend(de_data[set][:(len(de_data[set]) // 10) + 1])
    es_test = []
    es_train = [[], [], [], [], [], [], [], [], [], []]
    for set in range(10):
        es_train[set] = es_data[set][len(es_data[set]) // 10:]
        es_test.extend(es_data[set][:(len(es_data[set]) // 10) + 1])

    clcs_set = []
    for set in range(10):
        print("Training set", set + 1)
        file_name = f"all_lang_{str(set + 1)}.clcs"
        svc = None
        if os.path.isfile(file_name):
            print("File found")
            svc = pickle.load(open(file_name, "rb"))
            print("loading done")
        else:
            print("no File found")
            train = en_train[set] + de_train[set] + es_train[set]
            preproc = [preprocessing.lower]
            svc = CrossLingualContendScoring(train, preproc)
            pickle.dump(svc, open(file_name, "wb"))  # save the svm to file
            print("Training Done!")
        clcs_set.append(svc)
    svc = Combined_CLCS(clcs_set, lambda d: d.set-1)

    print("English:")
    en_val = validate(svc, en_test)
    print_validation(en_val)
    print("Spanish:")
    es_val = validate(svc, es_test)
    print_validation(es_val)
    print("German:")
    de_val = validate(svc, de_test)
    print_validation(de_val)
    print("")
    print("Total:")
    t_val = validate(svc, de_test + es_test + en_test)
    print_validation(t_val)
    print("")


if __name__ == "__main__":
    main_eval_total()
