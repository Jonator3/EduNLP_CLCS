import os
from typing import List

import pandas

import data
import nltk


nltk.download('punkt')


char_only_langs = ["zh"]
prompts = ["1", "2", "10"]


def tuple_to_bracketless_str(tup, padding=""):
    return padding.join(tup)


def concat(list):
    out = list[0]
    for L in list[1:]:
        out += L
    return out


def get_base_lang(dataset: str):
    langs = os.listdir("./data/"+dataset)
    base = dataset.replace("ASAP_", "")
    if langs.__contains__(base):
        return base
    else:
        return "en"


def count(list: List[str]):
    list = list.copy()
    list.sort()
    counts = {}
    current_str = list[0]
    last_change = 0
    for i in range(1, len(list)):
        S = list[i]
        if S != current_str:
            cnt = i - last_change
            counts[current_str] = cnt
            current_str = S
            last_change = i
    cnt = len(list) - last_change
    counts[current_str] = cnt

    return counts


def analyse(df, output_path):
    texts = [t for t in df["text"]]

    # char
    char_texts = [list(t) for t in texts]
    for n in range(1, 4):
        total_text = [tuple_to_bracketless_str(tup) for tup in concat([list(nltk.ngrams(t, n)) for t in char_texts])]
        counts = count(total_text)
        output = {
            "text": [],
            "count": []
        }
        for key in counts.keys():
            output["text"].append(key)
            output["count"].append(counts.get(key))
        out_df = pandas.DataFrame(output)
        out_df.to_csv(output_path+"/char_"+str(n)+"-gram.tsv", sep="\t", index=False)


    # word token
    if not char_only_langs.__contains__(lang):
        token_texts = [nltk.tokenize.word_tokenize(t) for t in texts]
        for n in range(1, 4):
            total_text = [tuple_to_bracketless_str(tup, " ") for tup in concat([list(nltk.ngrams(t, n)) for t in token_texts])]
            counts = count(total_text)
            output = {
                "text": [],
                "count": []
            }
            for key in counts.keys():
                output["text"].append(key)
                output["count"].append(counts.get(key))
            out_df = pandas.DataFrame(output)
            out_df.to_csv(output_path+"/word_token_"+str(n)+"-gram.tsv", sep="\t", index=False)


if __name__ == "__main__":
    datasets = [(ds, get_base_lang(ds)) for ds in os.listdir("./data")]
    for ds, lang in datasets:
        if lang != "en":
            datasets.append((ds, "en"))
    for ds, lang in datasets:
        for prmpt in prompts:
            df = data.load_data(ds, lang, prmpt)
            try:
                os.makedirs("./result/data_analysis/"+ds+"/prompt_"+prmpt+"/"+lang)
            except FileExistsError:
                pass
            analyse(df, "./result/data_analysis/"+ds+"/prompt_"+prmpt+"/"+lang)
