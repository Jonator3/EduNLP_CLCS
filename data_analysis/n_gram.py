import os
from typing import List

import pandas

import data
import nltk


nltk.download('punkt')


char_only_langs = ["zh"]
prompts = ["1", "2", "10"]


def tuple_to_bracketless_str(tup, padding=""):
    s = padding.join(tup)
    if s.startswith("'"):
        s = " " + s
    return s


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


def count_text(text_list):
    counts = count(text_list)
    output = {
        "text": [],
        "count": []
    }
    for key in counts.keys():
        if counts.get(key) is not None:
            if counts.get(key) > 0:
                output["text"].append(key)
                output["count"].append(counts.get(key))
    out_df = pandas.DataFrame(output)
    return out_df


def colum_rename(df, old, new):
    map = {}
    for col in df.columns:
        if col == old:
            map[col] = new
        else:
            map[col] = col
    return df.rename(columns=map)


def analyse(df, output_path):
    texts = [t for t in df["text"]]
    scores = list(set([s for s in df["score"]]))
    scores.sort()

    score_texts = {}
    for S in scores:
        S_df = df[df["score"] == S]
        score_texts[S] = [t for t in S_df["text"]]

    # char
    char_texts = [list(t) for t in texts]
    for n in range(1, 4):
        total_text = [tuple_to_bracketless_str(tup) for tup in concat([list(nltk.ngrams(t, n)) for t in char_texts])]
        out_df = count_text(total_text)
        out_df = colum_rename(out_df, "count", "total_count")
        for S in scores:
            S_texts = [tuple_to_bracketless_str(tup) for tup in concat([list(nltk.ngrams(list(t), n)) for t in score_texts.get(S)])]
            Sc_df = colum_rename(count_text(S_texts), "count", "s"+str(S)+"_count")
            out_df = out_df.join(Sc_df.set_index('text'), on='text')
            out_df["s"+str(S)+"_count"].fillna(0, inplace=True)
        out_df = out_df.sort_values(by="total_count", ascending=False).reset_index(drop=True)
        out_df.to_csv(output_path+"/char_"+str(n)+"-gram.tsv", sep="\t", index=False)


    # word token
    if not char_only_langs.__contains__(lang):
        token_texts = [nltk.tokenize.word_tokenize(t) for t in texts]
        for n in range(1, 4):
            total_text = [tuple_to_bracketless_str(tup, " ") for tup in concat([list(nltk.ngrams(t, n)) for t in token_texts])]
            out_df = count_text(total_text)
            out_df = colum_rename(out_df, "count", "total_count")
            for S in scores:
                S_texts = [tuple_to_bracketless_str(tup, " ") for tup in concat([list(nltk.ngrams(nltk.tokenize.word_tokenize(t), n)) for t in score_texts.get(S)])]
                Sc_df = colum_rename(count_text(S_texts), "count", "s"+str(S)+"_count")
                out_df = out_df.join(Sc_df.set_index('text'), on='text')
                out_df["s"+str(S)+"_count"].fillna(0, inplace=True)
            out_df = out_df.sort_values(by="total_count", ascending=False).reset_index(drop=True)
            out_df.to_csv(output_path+"/word_token_"+str(n)+"-gram.tsv", sep="\t", index=False)


def run():
    datasets = [(ds, get_base_lang(ds)) for ds in os.listdir("./data")]
    for ds, lang in datasets:
        if lang != "en":
            datasets.append((ds, "en"))
    datasets.sort(key=lambda x: x[0])
    for ds, lang in datasets:
        for prmpt in prompts:
            df = data.load_data(ds, lang, prmpt)
            try:
                os.makedirs("./result/data_analysis/" + ds + "/n-gram_count/prompt_" + prmpt + "/" + lang)
            except FileExistsError:
                pass
            analyse(df, "./result/data_analysis/" + ds + "/n-gram_count/prompt_" + prmpt + "/" + lang)
            print(ds + "/" + lang, "prompt_" + prmpt, "done!")


if __name__ == "__main__":
    run()