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


def stuff_str(s, l, attach_left=False, stuff_char=" "):
    while len(s) < l:
        if attach_left:
            s = stuff_char + s
        else:
            s += stuff_char
    return s


def print_validation(gold, predict, name1="Gold", name2="Prediction", stuff_len=5):
    mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(len(gold)):
        g = gold[i]
        p = predict[i]
        mat[g][p] += 1
    name1 = stuff_str(name1, max(stuff_len, 4))
    print("")
    print(name1, "|" + stuff_str(stuff_str(name2, ((stuff_len*4)+3-len(name2))/2, True), (stuff_len*4)+3)+"|")
    print(stuff_str("", len(name1)+1)+"|"+stuff_str("0", stuff_len)+"|"+stuff_str("1", stuff_len)+"|"+stuff_str("2", stuff_len)+"|"+stuff_str("3", stuff_len)+"|")
    for g, pre in enumerate(mat):
        s = stuff_str(str(g), len(name1)+1) + "|"
        for i in range(4):
            s += stuff_str(str(pre[i]), stuff_len, True) + "|"
        print(s)
    print("--------------------------------------")
    kappa = round(cohen_kappa_score(gold, predict, weights="quadratic"), 3)
    acc = round(accuracy_score(gold, predict), 3)
    print("quadratic_kappa =", kappa)
    print("accuracy =", acc)
    print("")
    return mat, kappa, acc


def main(lang, trainset, kfold=0, testset=None, name1="trainset", name2="testset"):
    preproc = [preprocessing.lower]
    result = [None, None, None, None, None, None, None, None, None, None]

    for prompt in [0, 1, 9]:

        print("\n\nPrompt:", prompt + 1)

        if kfold <= 0:
            svc = LogResClassifier(preproc, lang)
            svc.train(trainset[prompt])
            gold, predict = validate(svc, testset[prompt])
            print("")
            print(name1+">"+lang+" - "+name2+">"+lang)
            result[prompt] = print_validation(gold, predict)

        else:
            svc = LogResClassifier(preproc, lang)
            gold, predict = svc.train(trainset[prompt], kfold=kfold)
            print("")
            print(name1+">"+lang+"  (K-Fold="+str(kfold)+")")
            result[prompt] = print_validation(gold, predict)

    return result



if __name__ == "__main__":
    #  main(ignore_en_only_prompt=True, subset_passes=15, preproc=[preprocessing.lower])
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--k-fold", type=int, default=0, help="Set ratio for K-Fold. 0 will be no K-Fold.")
    argparser.add_argument("--subset", type=int, nargs=2, default=(0, 0), help="Set size and count of subsets to be used. 0 will be Off.", metavar=("size", "count"))
    argparser.add_argument("--testset", type=str, default="", help="Set path of the testset used to validate. Must be given if K-Fold is off.", metavar="filepath")
    argparser.add_argument("trainset", type=str, help="Set path of the trainingsset used.")
    argparser.add_argument("lang", type=str, help="Set Language to be used.", choices=("og", "en", "de", "es"))

    args = argparser.parse_args(sys.argv[1:])

    kfold = args.k_fold
    trainset_path = args.trainset
    testset_path = args.testset
    lang = args.lang
    subset_size, subset_count = args.subset

    if subset_size > 0 and subset_count > 0:
        trainset = get_subsets(separate_set(load_data(trainset_path)), subset_size, subset_count)
        testset = None
        if kfold == 0:
            testset = separate_set(load_data(testset_path))
        total = []
        for i in range(10):
            total.append([0, 0, 0])
        for subset in trainset:
            res = main(lang, subset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1])
            for i, p in enumerate(res):
                for i2 in range(1, 3):
                    total[i][i2] += p[i2]
        for i in range(len(total)):
            for i2 in range(1, 3):
                    total[i][i2] = total[i][i2]/subset_count
        print("\nMean result:")
        for i, t in enumerate(total):
            print("Prompt", i+1, ":\tkappa:", round(t[1], 3), ",\t accuracy:", round(t[2], 3))
    else:  # No subsets used
        trainset = separate_set(load_data(trainset_path))
        testset = None
        if kfold == 0:
            testset = separate_set(load_data(testset_path))
        main(lang, trainset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1])

