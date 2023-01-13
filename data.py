
from typing import List
import csv
import pandas as pd


dataset_base_path = "./data"


def get_fitting_index(key: str, arr: List[str]) -> int:
    if key.isdigit():
        return int(key)
    elif arr is None:
        raise ValueError("can`t find colum: " + key + " without Header! -> please use colum index instead.")
    elif arr.__contains__(key):  # look if the Key is in the Header
        return arr.index(key)
    elif [k.lower() for k in arr].__contains__(key.lower()):  # look if the Key is in the Header, ignoring upper case.
        return [k.lower() for k in arr].index(key.lower())
    else:  # there is no entry for the Key.
        raise ValueError("colum: " + key + " not found!")


def load_data(dataset, lang, prompt, use_test=False):
    name = "en"
    if dataset == "ASAP_orig":
        name = "orig_"
        if use_test:
            name += "test"
        else:
            name += "train"
    else:
        name = dataset.replace("ASAP_", "")

    data_path = dataset_base_path + "/" + dataset + "/" + name + "_prompt" + prompt + "_gold.tsv"
    text_path = dataset_base_path + "/" + dataset + "/" + lang + "/" + name + "_answers_" + lang + "_prompt" + prompt + ".tsv"

    df = to_df(data_path)
    other = to_df(text_path)

    df = df.join(other.set_index('id'), on='id')
    return df


def to_df(input_path):
    delimiter = ","
    if input_path.endswith(".tsv"):
        delimiter = "\t"
    reader = csv.reader(open(input_path, "r"), delimiter=delimiter, lineterminator="\n")

    colums = reader.__next__()
    data = [row for row in reader]

    df = {}
    for i, col in enumerate(colums):
        df[col.lower()] = [d[i] for d in data]

    return pd.DataFrame(df)


def balance_set(dataset: pd.DataFrame) -> pd.DataFrame:
    scores = dataset["score"].drop_duplicates().reset_index(drop=True)
    min_len = min(*[c for c in dataset["score"].value_counts()])

    #  This will reduce the count of Entry`s with Score X to the highest number so that all Scores have the same count.
    return pd.concat([D[D.index < min_len] for D in [dataset[dataset["score"] == S] for S in scores]], ignore_index=True)


def get_subsets(base_set: pd.DataFrame, length: int, count=10, balance=False) -> List[pd.DataFrame]:
    subsets = []
    for n in range(count):
        subset = []
        for prompt in base_set["prompt"].drop_duplicates().reset_index(drop=True):
            dataset = base_set[base_set["prompt"] == prompt]  # filter for Prompt
            dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle Dataset
            if balance:
                dataset = balance_set(dataset)
            subset.append(dataset[dataset.index < length])
        subsets.append(pd.concat(subset, ignore_index=True))
    return subsets


def get_limited_set(base_set: pd.DataFrame, length: int, balance=True) -> pd.DataFrame:
    return get_subsets(base_set, length, 1, balance)[0]
