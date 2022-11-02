
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing as preproc
import train_bert


class BertClassifier(object):

    def __init__(self, preprocessing=[], lang="en"):
        self.preprocessing = preprocessing  # TODO apply preprocessing
        self.model = None
        self.tokenizer = None
        self.lang = lang

    def train(self, trainingset, kfold=0, verbose=False):
        if kfold > 0:
            gold = []
            predict = []
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 1
            for _, test_index in kf.split(trainingset):  # All Indexes that are not in test will be used in train
                (_, train_df), (_, test_df) = trainingset.groupby(lambda d_i: test_index.__contains__(d_i))

                self.model, self.tokenizer = train_bert.train(train_df, "text", "score")

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
            self.model, self.tokenizer = train_bert.train(trainingset, "text", "score")
            return [], []

    def predict(self, text: str) -> int:
        return train_bert.predict(self.model, self.tokenizer, text)

