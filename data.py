
from typing import List
import csv
import pandas as pd


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


def load_data(input_path: str, id_col="id", prompt_col="prompt", score_col="score", text_col="text", has_head=True) -> pd.DataFrame:
    delimiter = ","
    if input_path.endswith(".tsv"):
        delimiter = "\t"
    reader = csv.reader(open(input_path, "r"), delimiter=delimiter)
    head = None
    if has_head:
        head = reader.__next__()
    id_index = get_fitting_index(id_col, head)
    prompt_index = get_fitting_index(prompt_col, head)
    score_index = get_fitting_index(score_col, head)
    text_index = get_fitting_index(text_col, head)

    data = [row for row in reader]  # read the file
    data = [(d[id_index], d[prompt_index], d[score_index], d[text_index]) for d in data]  # filter columns
    data = pd.DataFrame.from_records(data, columns=['id', 'prompt', 'score', 'text'])  # make pandas.DataFrame

    return data


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
