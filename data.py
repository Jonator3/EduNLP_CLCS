from typing import List
import random
import csv


# TODO rewrite Datastructure with pandas
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

    def get_text(self, lang="og"):
        if lang == "og":
            return self.og_text
        elif lang == "de":
            return self.de_text
        elif lang == "en":
            return self.en_text
        elif lang == "es":
            return self.es_text


def load_data(input_path: str) -> List[CrossLingualDataEntry]:  # TODO use pandas and generalize
    data = []
    reader = csv.reader(open(input_path, "r"))
    reader.__next__()  # skip head
    for row in reader:
        entry = CrossLingualDataEntry(row[0], row[4], row[1], row[2], row[3], row[5], row[6], row[7])
        data.append(entry)
    return data


def separate_set(dataset: List[CrossLingualDataEntry]):  # TODO generalize for use with pandas
    output = [[], [], [], [], [], [], [], [], [], []]
    for d in dataset:
        output[d.set - 1].append(d)
    return output


def balance_set(dataset: List[CrossLingualDataEntry]):  # TODO generalize for use with pandas
    data = {}
    for D in dataset:
        if data.get(D.gold_score) is None:  # add new list for score
            data[D.gold_score] = []
        data.get(D.gold_score).append(D)
    scores = data.keys()
    min_len = -1
    for s in scores:
        l = len(data.get(s))  # count of datapoints with score s
        if min_len > l or min_len < 0:
            min_len = l
    balanced_dataset = []
    for s in scores:
        s_set = data.get(s)[0:min_len]
        balanced_dataset += s_set
    return balanced_dataset


def get_subsets(base_set: List[CrossLingualDataEntry], length, count=10, balance=False):  # TODO generalize for use with pandas
    subsets = []
    for n in range(count):
        subsets.append([])
        for prompt in set([d.set for d in base_set]):
            dataset = [d for d in base_set if d.set == prompt]
            random.shuffle(dataset)
            if balance:
                data = {}
                for D in base_set:
                    if data.get(D.gold_score) is None:  # add new list for score
                        data[D.gold_score] = []
                    data.get(D.gold_score).append(D)
                scores = data.keys()
                min_len = length//len(scores)
                for sc in scores:
                    l = len(data.get(sc))  # count of datapoints with score s
                    if min_len > l:
                        min_len = l
                dataset = []
                for sc in scores:
                    s = data.get(sc)[0:min_len]
                    dataset += s
            subsets[n] += dataset[:length]
    return subsets


def get_langgraber(lang):  # TODO rewrite for pandas
    langgraber = lambda x: (x.og_text, x.lang)
    if lang == "en":
        langgraber = lambda x: (x.en_text, "en")
    if lang == "de":
        langgraber = lambda x: (x.de_text, "de")
    if lang == "es":
        langgraber = lambda x: (x.es_text, "es")
    return langgraber
