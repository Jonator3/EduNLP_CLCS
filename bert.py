from typing import List

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import data
import preprocessing
import train_bert


Train_Batch_Size = 16


def separate_df(test_index: List[int], df):
    train_df = None
    test_df = None
    for v, v_df in df.groupby(lambda d_i: test_index.__contains__(d_i)):
        if type(v) != bool:
            raise ValueError("Impossible outcome, v should only be boolean!")
        if v:
            test_df = v_df
        else:
            train_df = v_df
    return train_df, test_df


class BertClassifier(object):

    def __init__(self, preproc=[], lang="en"):
        self.preprocessing = preprocessing.compose(*preproc)
        self.model = None
        self.tokenizer = None
        self.lang = lang

    def train(self, trainingset, kfold=0, verbose=False):
        trainingset = data.apply_to_text(trainingset, self.preprocessing)
        if kfold > 0:
            trainingset = trainingset.reset_index(drop=True)  # ensure sequential index starting by 0
            gold = []
            predict = []
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 1
            for _, test_index in kf.split(trainingset):  # All Indexes that are not in test will be used in train
                train_df, test_df = separate_df(test_index, trainingset)
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                self.model, self.tokenizer = train_bert.train(train_df, "text", "score", Train_Batch_Size)

                p = []
                if trainingset.columns.__contains__("text2"):
                    p = [self.predict(t) for t in test_df["text2"]]
                else:
                    p = [self.predict(t) for t in test_df["text"]]
                g = [s for s in test_df["score"]]
                gold += g
                predict += p
                if verbose:
                    print(f"Fold no. {i}")
                    print(f"Accuracy = {accuracy_score(g, p)}")
                    print(f"Kappa = {cohen_kappa_score(g, p, weights='quadratic')}")
                i += 1
            return gold, predict
        else:
            self.model, self.tokenizer = train_bert.train(trainingset, "text", "score", Train_Batch_Size)
            return [], []

    def predict(self, text: str) -> str:
        return str(train_bert.predict(self.model, self.tokenizer, text))

