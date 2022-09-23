import argparse
import sys

from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing
from data import *
from log_res import LogResClassifier


def validate(classifier, dataset: List[CrossLingualDataEntry]):
    predict = []
    gold = []
    for data in dataset:
        p = classifier.predict(data)
        g = data.gold_score
        predict.append(p)
        gold.append(g)
    return gold, predict


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

        svc = LogResClassifier(preproc, "og")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN-EN")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = LogResClassifier(preproc, "og")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300-EN300 (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "og")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk-EN_shk (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "og")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE-DE (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "og")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES-ES (KFold)")
        print_validation(gold, predict)


        print("=== Translate Both ===")

        # Translated EN
        svc = LogResClassifier(preproc, "de")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN>DE-EN>DE")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        svc.train(en_train[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("EN>ES-EN>ES")
        print_validation(gold, predict)

        # Translated EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = LogResClassifier(preproc, "de")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300>DE-EN300>DE (KFold)")
        print_validation(gold, predict)

        gold = []
        predict = []
        for Dset in en_train_300:
            svc = LogResClassifier(preproc, "es")
            g, p = svc.train(Dset[set], kfold=True)
            gold += g
            predict += p
        print("")
        print("EN300>ES-EN300>ES (KFold)")
        print_validation(gold, predict)

        # Translated EN_shk
        svc = LogResClassifier(preproc, "de")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk>DE-EN_shk>DE (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        gold, predict = svc.train(en_shk[set], kfold=True)
        print("")
        print("EN_shk>ES-EN_shk>ES (KFold)")
        print_validation(gold, predict)

        # Translated DE
        svc = LogResClassifier(preproc, "en")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE>EN-DE>EN (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        gold, predict = svc.train(de_test[set], kfold=True)
        print("")
        print("DE>ES-DE>ES (KFold)")
        print_validation(gold, predict)

        # Translate ES
        svc = LogResClassifier(preproc, "en")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES>EN-ES>EN (KFold)")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "de")
        gold, predict = svc.train(es_test[set], kfold=True)
        print("")
        print("ES>DE-ES>DE (KFold)")
        print_validation(gold, predict)


        print("=== Translate Train ===")

        # Train EN
        svc = LogResClassifier(preproc, "de")
        svc.train(en_train[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN>DE-DE")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        svc.train(en_train[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN>ES-ES")
        print_validation(gold, predict)

        # Train EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = LogResClassifier(preproc, "de")
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
            svc = LogResClassifier(preproc, "es")
            svc.train(Dset[set])
            g, p = validate(svc, es_test[set])
            gold += g
            predict += p
        print("")
        print("EN300>ES-ES")
        print_validation(gold, predict)

        # Train EN_shk
        svc = LogResClassifier(preproc, "de")
        svc.train(en_shk[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN_shk>DE-DE")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        svc.train(en_shk[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN_shk>ES-ES")
        print_validation(gold, predict)

        # Train DE
        svc = LogResClassifier(preproc, "en")
        svc.train(de_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        svc.train(de_test[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("DE>ES-ES")
        print_validation(gold, predict)

        # Train ES
        svc = LogResClassifier(preproc, "en")
        svc.train(es_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("ES>EN-EN")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "de")
        svc.train(es_test[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("ES>DE-ES")
        print_validation(gold, predict)


        print("=== Translate Test ===")

        # Train EN
        svc = LogResClassifier(preproc, "en")
        svc.train(en_train[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN-DE>EN")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "en")
        svc.train(en_train[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN-ES>EN")
        print_validation(gold, predict)

        # Train EN300
        gold = []
        predict = []
        for Dset in en_train_300:
            svc = LogResClassifier(preproc, "en")
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
            svc = LogResClassifier(preproc, "en")
            svc.train(Dset[set])
            g, p = validate(svc, es_test[set])
            gold += g
            predict += p
        print("")
        print("EN300-ES>EN")
        print_validation(gold, predict)

        # Train EN_shk
        svc = LogResClassifier(preproc, "en")
        svc.train(en_shk[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("EN_shk>DE-DE")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "en")
        svc.train(en_shk[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("EN_shk>ES-ES")
        print_validation(gold, predict)

        # Train DE
        svc = LogResClassifier(preproc, "de")
        svc.train(de_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("DE>EN-EN")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "de")
        svc.train(de_test[set])
        gold, predict = validate(svc, es_test[set])
        print("")
        print("DE>ES-ES")
        print_validation(gold, predict)

        # Train ES
        svc = LogResClassifier(preproc, "es")
        svc.train(es_test[set])
        gold, predict = validate(svc, en_test[set])
        print("")
        print("ES>EN-EN")
        print_validation(gold, predict)

        svc = LogResClassifier(preproc, "es")
        svc.train(es_test[set])
        gold, predict = validate(svc, de_test[set])
        print("")
        print("ES>DE-DE")
        print_validation(gold, predict)


if __name__ == "__main__":
    #  main(ignore_en_only_prompt=True, subset_passes=15, preproc=[preprocessing.lower])
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--k-fold", type=int, default=0, help="Set ratio for K-Fold. 0 will be no K-Fold.")
    argparser.add_argument("--subset", type=int, nargs=2, default=(0, 0), help="Set size and count of subsets to be used. 0 will be Off.", metavar=("size", "count"))
    argparser.add_argument("--testset", type=str, default="", help="Set path of the testset used to validate. Must be given if K-Fold is off.", metavar="filepath")
    argparser.add_argument("trainset", type=str, help="Set path of the trainingsset used.")

    args = argparser.parse_args(sys.argv[1:])
    print(args)
