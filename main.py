import argparse
import os.path
import sys
from datetime import datetime

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import preprocessing
from data import load_data, get_subsets, balance_set
from log_res import LogResClassifier
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


def main(lang, trainset: pd.DataFrame, kfold=0, testset: pd.DataFrame = None, name1="trainset", name2="testset", preproc=[], print_result=True):
    print(name1 + " -> " + lang + " -- " + name2 + " -> " + lang)
    prompts = trainset["prompt"].drop_duplicates().reset_index(drop=True)
    result = {
        "Model": ["LogRes"],  # TODO change for Bert
        "trainset": [name1],
        "train_col": [lang],
        "testset": [name2],
        "K-Fold": [kfold]
    }

    for prompt in prompts:

        trainset_p = trainset[trainset["prompt"] == prompt]  # filter Trainset for specific Prompt
        if print_result:
            print("\n\nPrompt:", prompt)

        if kfold <= 0:
            testset_p = testset[testset["prompt"] == prompt]
            classifier = LogResClassifier(preproc)
            classifier.train(trainset_p)
            gold, predict = validate(classifier, testset_p)
            res = make_validation_table(gold, predict)
            result["QWK_"+prompt] = [res[1]]
            if print_result:
                print("")
                print_validation(*res)
        else:
            classifier = LogResClassifier(preproc)
            gold, predict = classifier.train(trainset_p, kfold=kfold)
            res = make_validation_table(gold, predict)
            result["QWK_"+prompt] = [res[1]]
            if print_result:
                print("")
                print(name1+">"+lang+"  (K-Fold="+str(kfold)+")")
                print("")
                print_validation(*res)
    qwk = 0.0
    for prompt in prompts:
        qwk += result.get("QWK_"+prompt)[0]
    result["QWK_Mean"] = [round(qwk / len(prompts), 3)]
    result["Timestamp"] = [datetime.now().strftime("%Y:%m:%d-%H:%M")]
    return pd.DataFrame(result)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    # TODO add arguments to save Model via Pickle.
    argparser.add_argument("--lowercase", default=False, action='store_true', help="Add lowercase to the preprocessing.")
    argparser.add_argument("--k-fold", type=int, default=0, help="Set ratio for K-Fold. 0 will be no K-Fold.")
    argparser.add_argument("--balance", default=False, action='store_true', help="Enable balancing of the trainset.")
    argparser.add_argument("--subset", type=int, nargs=2, default=(0, 0), help="Set size and count of subsets to be used. 0 will be Off.", metavar=("size", "count"))
    argparser.add_argument("--output", type=str, default="", help="Set path of the output CSV-File", metavar="filepath")
    argparser.add_argument("--testset", type=str, default="", help="Set path of the testset used to validate. Must be given if K-Fold is off.", metavar="filepath")
    argparser.add_argument("--testset_id", type=str, default="id", help="Set Colum Name or Index for the Entry ID.", metavar="colum name/index")
    argparser.add_argument("--testset_prompt", type=str, default="essayset", help="Set Colum Name or Index for the Prompt.", metavar="colum name/index")
    argparser.add_argument("--testset_score", type=str, default="score", help="Set Colum Name or Index for the Goldscore.", metavar="colum name/index")
    argparser.add_argument("--testset_text", type=str, default="originaltext", help="Set Colum Name or Index for the Text to score.", metavar="colum name/index")
    argparser.add_argument("--testset_no_header", type=bool, default=False, help="Indicate that the CSV-File has no Header")
    argparser.add_argument("trainset", type=str, help="Set path of the trainingsset used.")
    argparser.add_argument("--trainset_id", type=str, default="id", help="Set Colum Name or Index for the Entry ID.", metavar="colum name/index")
    argparser.add_argument("--trainset_prompt", type=str, default="essayset", help="Set Colum Name or Index for the Prompt.", metavar="colum name/index")
    argparser.add_argument("--trainset_score", type=str, default="score", help="Set Colum Name or Index for the Goldscore.", metavar="colum name/index")
    argparser.add_argument("--trainset_text", type=str, default="originaltext", help="Set Colum Name or Index for the Text to score.", metavar="colum name/index")
    argparser.add_argument("--trainset_no_header", type=bool, default=False, help="Indicate that the CSV-File has no Header")

    args = argparser.parse_args(sys.argv[1:])

    kfold = args.k_fold
    output_path = args.output
    trainset_path = args.trainset
    trainset_conf = (args.trainset_id, args.trainset_prompt, args.trainset_score, args.trainset_text, not args.trainset_no_header)
    testset_path = args.testset
    testset_conf = (args.testset_id, args.testset_prompt, args.testset_score, args.testset_text, not args.testset_no_header)
    balance = args.balance
    lang = args.trainset_text
    subset_size, subset_count = args.subset
    preproc = []
    if args.lowercase:
        preproc.append(preprocessing.lower)

    trainset = load_data(trainset_path, *trainset_conf)
    testset = None
    result = None
    if kfold == 0:
        testset = load_data(testset_path, *testset_conf)

    if subset_size > 0 and subset_count > 0:
        trainset = get_subsets(trainset, subset_size, subset_count, balance)
        results = []
        for subset in trainset:
            res = main(lang, subset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1], print_result=False)
            results.append(res)
        results = pd.concat(results, ignore_index=True)

        mean = {}
        for key in results.keys():
            values = [v for v in results[key]]
            if key.startswith("QWK"):
                mean_val = sum(values)/len(values)
                mean[key] = mean_val
            else:
                mean[key] = values[0]
        mean = pd.DataFrame(mean)
        print(mean)
        result = mean

    else:  # No subsets used
        if balance:
            trainset = balance_set(trainset)
        result = main(lang, trainset, kfold, testset, trainset_path.split("/")[-1], testset_path.split("/")[-1])

    if output_path != "":  # if a path for the output is given, write to it.
        folder = "/".join(output_path.split("/")[:-1])
        if not os.path.exists(folder):  # output folder does not exist
            os.makedirs(folder)
        if os.path.exists(output_path):  # if Outputfile exists, append data.
            data = pd.read_csv(output_path)
            result = pd.concat([data, result], ignore_index=True)
        # TODO AH: also provide an option to save (pickle) the learnt model and store the predictions of the classifier (per item: id, raw answer text, gold, pred)
        result.to_csv(output_path, index=False)


