import argparse
import sys

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import preprocessing
from data import *  # TODO make nicer import (no *)
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


def stuff_str(s, l, attach_left=False, stuff_char=" "):
    while len(s) < l:
        if attach_left:
            s = stuff_char + s
        else:
            s += stuff_char
    return s


def make_validation_table(gold, predict):
    mat = confusion_matrix(gold, predict, labels=(0, 1, 2, 3))
    kappa = round(cohen_kappa_score(gold, predict, weights="quadratic"), 3)
    acc = round(accuracy_score(gold, predict), 3)

    return mat, kappa, acc


def print_validation(mat, kappa, acc, name1="Gold", name2="Prediction", stuff_len=5):
    name1 = stuff_str(name1, max(stuff_len, 4))
    print("")
    print(name1, "|" + stuff_str(stuff_str(name2, ((stuff_len*4)+3+len(name2))/2, True), (stuff_len*4)+3)+"|")
    print(stuff_str("", len(name1)+1)+"|"+stuff_str("0", stuff_len)+"|"+stuff_str("1", stuff_len)+"|"+stuff_str("2", stuff_len)+"|"+stuff_str("3", stuff_len)+"|")
    for g, pre in enumerate(mat):
        s = stuff_str(str(g), len(name1)+1) + "|"
        for i in range(4):
            s += stuff_str(str(pre[i]), stuff_len, True) + "|"
        print(s)
    print("--------------------------------------")

    print("quadratic_kappa =", kappa)
    print("accuracy =", acc)
    print("")


def main(lang, trainset, kfold=0, testset=None, name1="trainset", name2="testset", print_result=True):
    preproc = [preprocessing.lower]
    result = [None, None, None, None, None, None, None, None, None, None] # TODO AH: use pandas dataframes to read data

    print(name1 + " -> " + lang + " -- " + name2 + " -> " + lang)

    for prompt in [0, 1, 9]:

        print("\n\nPrompt:", prompt + 1)

        if kfold <= 0:
            classifier = LogResClassifier(preproc, lang)
            classifier.train(trainset[prompt])
            gold, predict = validate(classifier, testset[prompt])
            print("")
            result[prompt] = make_validation_table(gold, predict)
            if print_result: # TODO AH: also store results into an output file in a separate folder: e.g. per experiment train data, test data, classifier, parameters, timestamp, evaluation results
                print_validation(*result[prompt]) # TODO AH. alsonprovide an option to save (pickle) the lernt model and store the predictions of the classifier (per item: id, raw answer text, gold, pred)
        else:
            classifier = LogResClassifier(preproc, lang)
            gold, predict = classifier.train(trainset[prompt], kfold=kfold)
            print("")
            print(name1+">"+lang+"  (K-Fold="+str(kfold)+")")
            result[prompt] = make_validation_table(gold, predict)
            if print_result:
                print_validation(*result[prompt])
    return result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    # TODO add arguments to generalize Datastructure of the Input, aka tell pandas how to read the Input-File.
    # TODO add arguments for selection of preprocessing.
    # TODO add arguments to save result into File.
    # TODO add arguments to save Model via Pickle.
    argparser.add_argument("--k-fold", type=int, default=0, help="Set ratio for K-Fold. 0 will be no K-Fold.")
    argparser.add_argument("--balance", type=bool, default=False, help="Enable balancing of the trainset.")
    argparser.add_argument("--subset", type=int, nargs=2, default=(0, 0), help="Set size and count of subsets to be used. 0 will be Off.", metavar=("size", "count"))
    argparser.add_argument("--testset", type=str, default="", help="Set path of the testset used to validate. Must be given if K-Fold is off.", metavar="filepath")
    argparser.add_argument("trainset", type=str, help="Set path of the trainingsset used.")
    argparser.add_argument("lang", type=str, help="Set Language to be used.", choices=("og", "en", "de", "es"))

    args = argparser.parse_args(sys.argv[1:])

    kfold = args.k_fold
    trainset_path = args.trainset
    testset_path = args.testset
    lang = args.lang
    balance = args.balance
    subset_size, subset_count = args.subset

    trainset = separate_set(load_data(trainset_path))
    testset = None
    if kfold == 0:
        testset = separate_set(load_data(testset_path))

    if subset_size > 0 and subset_count > 0:
        trainset = get_subsets(trainset, subset_size, subset_count, balance)
        total = []
        for i in range(10):
            total.append([0, 0, 0])
        for subset in trainset:
            res = main(lang, subset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1], False)
            for i, p in enumerate(res):
                for i2 in range(1, 3):
                    if p[i2] is not None:
                        total[i][i2] += p[i2]
        for i in range(len(total)):
            for i2 in range(1, 3):
                    total[i][i2] = total[i][i2]/subset_count
        print("\nMean result:")
        for i, t in enumerate(total):
            print("Prompt", i+1, ":\tkappa:", round(t[1], 3), ",\t accuracy:", round(t[2], 3))
    else:  # No subsets used
        if balance:
            trainset = balance_set(trainset)
        main(lang, trainset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1])

