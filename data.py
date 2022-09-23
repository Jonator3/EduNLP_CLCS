from typing import List
import random
import csv


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


def load_data(input_path: str) -> List[CrossLingualDataEntry]:
    data = []
    reader = csv.reader(open(input_path, "r"))
    reader.__next__()  # skip head
    for row in reader:
        entry = CrossLingualDataEntry(row[0], row[4], row[1], row[2], row[3], row[5], row[6], row[7])
        data.append(entry)
    return data



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
