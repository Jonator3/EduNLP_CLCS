import argparse
import math
import os.path
import pickle
import sys
from datetime import datetime

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import bert
import preprocessing
from data import load_data, get_subsets, balance_set
from log_res import LogResClassifier, LogResNCharClassifier
from bert import BertClassifier
import pandas as pd


def validate(classifier, dataset: pd.DataFrame):
    predict = [classifier.predict(t) for t in dataset["text"]]
    gold = [s for s in dataset["score"]]
    return gold, predict


def get_average(list):
    return sum(list)/len(list)


def stuff_str(s: str, length: int, attach_left=False, stuff_char=" ") -> str:
    while len(s) < length:
        if attach_left:
            s = stuff_char + s
        else:
            s += stuff_char
    return s


def make_validation_table(gold, predict):
    mat = confusion_matrix(gold, predict, labels=("0", "1", "2", "3"))
    kappa = round(cohen_kappa_score(gold, predict, weights="quadratic"), 3)
    acc = round(accuracy_score(gold, predict), 3)

    return mat, kappa, acc


def print_validation(mat, kappa, acc, name1="Gold", name2="Prediction", stuff_len=5):
    name1 = stuff_str(name1, max(stuff_len, 4))
    print("")
    print(name1, "|" + stuff_str(stuff_str(name2, ((stuff_len*4)+3+len(name2))//2, True), (stuff_len*4)+3)+"|")
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


Classifier_Models = {
    "logres": LogResClassifier,
    "bert": BertClassifier,
    "logres_char": LogResNCharClassifier
}


def main(trainset: pd.DataFrame, kfold=0, testset: pd.DataFrame = None, name1="trainset", lang1="train-language", name2="testset", lang2="test-language", preproc=[], print_result=True, save_model=None, model="logres"):
    if print_result:
        print(name1 + " --> " + name2)
    prompts = trainset["prompt"].drop_duplicates().reset_index(drop=True)
    result = {
        "Timestamp": [datetime.now().strftime("%Y:%m:%d-%H:%M")],
        "Model": [model],
        "trainset": [name1],
        "train-language": [lang1],
        "testset": [name2],
        "test-language": [lang2],
        "K-Fold": [kfold]
    }

    for prompt in prompts:

        trainset_p = trainset[trainset["prompt"] == prompt]  # filter Trainset for specific Prompt
        if print_result:
            print("\n\nPrompt:", prompt)

        classifier = Classifier_Models.get(model)(preproc)  # Make Instance of Classifier Model of choose.
        if kfold <= 0:
            testset_p = testset[testset["prompt"] == prompt]
            classifier.train(trainset_p)
            gold, predict = validate(classifier, testset_p)
            res = make_validation_table(gold, predict)
            result["QWK_"+prompt] = [res[1]]
            if print_result:
                print("")
                print_validation(*res)
        else:
            gold, predict = classifier.train(trainset_p, kfold=kfold)
            res = make_validation_table(gold, predict)
            result["QWK_"+prompt] = [res[1]]
            if print_result:
                print("")
                print(name1+"> (K-Fold="+str(kfold)+")")
                print("")
                print_validation(*res)
        if save_model is not None:  # path for saving the Models is given.
            filepath = save_model + "_" + prompt + ".pickle"
            folder = "/".join(filepath.split("/")[:-1])
            if not os.path.exists(folder):  # output folder does not exist
                os.makedirs(folder)
            pickle.dump(classifier, open(filepath, "bw+"))
    return pd.DataFrame(result)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--lowercase", default=False, action='store_true', help="Add lowercase to the preprocessing.")
    argparser.add_argument("--classifier", type=str, default="logres", help="Classifiermodel to be used.", metavar="model", choices=Classifier_Models.keys())
    argparser.add_argument("--bert_batch_size", type=int, default=16, help="Set Batch size for training Bert", metavar="size")
    argparser.add_argument("--k-fold", type=int, default=0, help="Set ratio for K-Fold. 0 will be no K-Fold.")
    argparser.add_argument("--balance", default=False, action='store_true', help="Enable balancing of the trainset.")
    argparser.add_argument("--subset", type=int, nargs=2, default=(0, 0), help="Set size and count of subsets to be used. 0 will be Off.", metavar=("size", "count"))
    argparser.add_argument("--output", type=str, default="", help="Set path of the output CSV-File", metavar="filepath")
    argparser.add_argument("--save_model", type=str, default=None, help="Enable and set path for saving the Models via Pickle.", metavar="path")
    argparser.add_argument("--testset", type=str, nargs=2, default=("", ""), help="Set testset and language used to validate. Must be given if K-Fold is off.", metavar=("Dataset", "lang"))
    argparser.add_argument("trainset", type=str, nargs=2, default=("", ""), help="Set trainset and language.", metavar=("Dataset", "lang"))
    argparser.add_argument("prompt", type=str, default="1", help="Prompt to be used.", metavar="prompt")

    args = argparser.parse_args(sys.argv[1:])

    kfold = args.k_fold
    output_path = args.output
    trainset_path, train_lang = args.trainset
    testset_path, test_lang = args.testset
    if testset_path == "":
        testset_path = trainset_path
        test_lang = train_lang
    prompt = args.prompt
    balance = args.balance
    save_model = args.save_model
    model = args.classifier
    subset_size, subset_count = args.subset
    bert.Train_Batch_Size = args.bert_batch_size
    preproc = []
    if args.lowercase:
        preproc.append(preprocessing.lower)

    trainset = load_data(trainset_path, train_lang, prompt)
    testset = None
    result = None
    if kfold == 0:
        testset = load_data(testset_path, test_lang, prompt, use_test=True)

    if subset_size > 0 and subset_count > 0:
        trainset = get_subsets(trainset, subset_size, subset_count, balance)
        results = []
        for i, subset in enumerate(trainset):
            print("Subset:\t", i, "/", len(trainset))
            sm = save_model
            if sm is not None:
                sm += "_" + stuff_str(str(i), math.floor(math.log10(len(trainset))), True, "0")  # add stuffed number of subset to filepath
            res = main(subset, kfold, testset, trainset_path + "-" + train_lang, train_lang, testset_path + "-" + test_lang, test_lang, print_result=False, save_model=sm, model=model)
            print("res", res)
            results.append(res)
        results = pd.concat(results, ignore_index=True)
        print("results", results)

        mean = {}
        for key in results.keys():
            values = [v for v in results[key]]
            if key.startswith("QWK"):
                mean_val = sum(values)/len(values)
                mean[key] = [mean_val]
                print(key, "=", mean_val)
            elif key == "trainset":
                mean[key] = [values[0] + "_subsets"]
            else:
                mean[key] = [values[0]]
        mean = pd.DataFrame(mean)
        print(mean)
        result = mean

    else:  # No subsets used
        if balance:
            trainset = balance_set(trainset)
        result = main(trainset, kfold, testset, trainset_path, train_lang, testset_path, test_lang, save_model=save_model, model=model)
#        result = main(trainset, kfold, testset, trainset_path + "-" + train_lang, testset_path + "-" + test_lang, save_model=save_model, model=model)

    if output_path != "":  # if a path for the output is given, write to it.
        folder = "/".join(output_path.split("/")[:-1])
        if not os.path.exists(folder):  # output folder does not exist
            os.makedirs(folder)
        if os.path.exists(output_path):  # if Outputfile exists, append data.
            data = pd.read_csv(output_path)
            #print(data)
            #print(result)
            result = pd.concat([data, result], ignore_index=True)
            print(result)
        result.to_csv(output_path, index=False)


