import os.path
import random
import sys
from typing import List

import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate, KFold
import pickle
import csv

import preprocessing


class CrossLingualDataEntry(object):

    def __init__(self, id, lang, set, gold_score, og_text, en_text, de_text, es_text):
        self.id = id
        self.lang = lang
        self.set = int(set)
        self.gold_score = int(gold_score)
        self.og_text = og_text
        self.en_text = en_text
        self.de_text = de_text
        self.es_text = es_text

    def __str__(self):
        return f"CLDE: {self.id} in set {self.set}: {self.gold_score}"


class CrossLingualContendScoring(object):

    def __init__(self, preprocessing=[], lang="en", vocabulary=None, use_LogRes=True):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        if use_LogRes:
            self.svc = LogisticRegression(max_iter=1000)
        else:
            self.svc = svm.SVC()
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
        entry = CrossLingualDataEntry(row[0], row[4], row[1], row[2], row[3], row[5], row[6], row[7])
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


def get_average(list):
    return sum(list)/len(list)


def print_validation(gold, predict, name1="Gold", name2="Prediction"):
    mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(len(gold)):
        g = gold[i]
        p = predict[i]
        mat[g][p] += 1
    print("")
    print(name1, "/\t", name2)
    print("\t|", "0\t\t|", "1\t\t|", "2\t\t|", "3")
    for g, pre in enumerate(mat):
        print(g, "\t|", pre[0], "\t|", pre[1], "\t|", pre[2], "\t|", pre[3])
    print("--------------------------------------")
    kappa = round(cohen_kappa_score(gold, predict, weights="quadratic"), 3)
    acc = round(accuracy_score(gold, predict), 3)
    print("quadratic_kappa =", kappa)
    print("accuracy =", acc)
    print("")
    return mat, kappa, acc


def separate_set(dataset: List[CrossLingualDataEntry]):
    output = [[], [], [], [], [], [], [], [], [], []]
    for d in dataset:
        output[d.set - 1].append(d)
    return output


def get_subsets(base_set, length, count=10):
    en_train_300 = []
    for n in range(count):
        en_train_300.append(base_set.copy())
        for i, set in enumerate(en_train_300[n]):
            s = set.copy()
            random.shuffle(s)
            en_train_300[n][i] = s[:length]
    return en_train_300


def main(ignore_en_only_prompt=True, subset_passes=10, preproc=[preprocessing.lower]):
    en_train = separate_set(load_data("data/en_train.csv"))
    en_test = separate_set(load_data("data/en_test.csv"))
    de_test = separate_set(load_data("data/de.csv"))
    es_test = separate_set(load_data("data/es.csv"))
    en_shk = separate_set(load_data("data/en_shk.csv"))

    en_train_300 = get_subsets(en_train, 300, subset_passes)


    for set in range(10):
        if ignore_en_only_prompt:
            if not [0, 1, 9].__contains__(set):  # will only run prompt 1, 2, 10
                continue

        print("\n\nSet", set + 1)

        print("=== Base-Line ===")

        svc = CrossLingualContendScoring(preproc, "og")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN-EN")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "og")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300-EN300 (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "og")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk-EN_shk (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "og")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE-DE (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "og")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES-ES (KFold)")
        print_validation(gold, predict)


        print("=== Translate Both ===")

        # Translated EN
        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN>DE-EN>DE")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN>ES-EN>ES")
        print_validation(gold, predict)

        # Translated EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "de")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300>DE-EN300>DE (KFold)")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "es")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300>ES-EN300>ES (KFold)")
        print_validation(gold, predict)

        # Translated EN_shk
        svc = CrossLingualContendScoring(preproc, "de")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk>DE-EN_shk>DE (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk>ES-EN_shk>ES (KFold)")
        print_validation(gold, predict)

        # Translated DE
        svc = CrossLingualContendScoring(preproc, "en")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE>EN-DE>EN (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE>ES-DE>ES (KFold)")
        print_validation(gold, predict)

        # Translate ES
        svc = CrossLingualContendScoring(preproc, "en")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES>EN-ES>EN (KFold)")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "de")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES>DE-ES>DE (KFold)")
        print_validation(gold, predict)


        print("=== Translate Train ===")

        # Train EN
        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(en_train[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN>DE-DE")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(en_train[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN>ES-ES")
        print_validation(gold, predict)

        # Train EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "de")
            svc.train(Dset[set])
            g, p = validate(svc, de_test[set])
            gold += g
            predict += p
        print("")
        print("EN300>DE-DE")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "es")
            svc.train(Dset[set])
            g, p = validate(svc, es_test[set])
            gold += g
            predict += p
        print("")
        print("EN300>ES-ES")
        print_validation(gold, predict)

        # Train EN_shk
        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(en_shk[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN_shk>DE-DE")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(en_shk[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN_shk>ES-ES")
        print_validation(gold, predict)

        # Train DE
        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(de_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(de_test[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("DE>ES-ES")
        print_validation(gold, predict)

        # Train ES
        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(es_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(es_test[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("DE>ES-ES")
        print_validation(gold, predict)


        print("=== Translate Test ===")

        # Train EN
        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(en_train[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN-DE>EN")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(en_train[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN-ES>EN")
        print_validation(gold, predict)

        # Train EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "en")
            svc.train(Dset[set])
            g, p = validate(svc, de_test[set])
            gold += g
            predict += p
        print("")
        print("EN300-DE>EN")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = CrossLingualContendScoring(preproc, "en")
            svc.train(Dset[set])
            g, p = validate(svc, es_test[set])
            gold += g
            predict += p
        print("")
        print("EN300-ES>EN")
        print_validation(gold, predict)

        # Train EN_shk
        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(en_shk[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN_shk>DE-DE")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(en_shk[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN_shk>ES-ES")
        print_validation(gold, predict)

        # Train DE
        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(de_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "es")
        svc.train(de_test[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("DE>ES-ES")
        print_validation(gold, predict)

        # Train ES
        svc = CrossLingualContendScoring(preproc, "en")
        svc.train(es_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = CrossLingualContendScoring(preproc, "de")
        svc.train(es_test[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("ES>DE-DE")
        print_validation(gold, predict)


if __name__ == "__main__":
    main(ignore_en_only_prompt=True, subset_passes=15, preproc=[preprocessing.lemmatize, preprocessing.lower])
