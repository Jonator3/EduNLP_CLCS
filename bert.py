
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing as preproc
from data import *
import train_bert


def prepare_data(dataset: List[CrossLingualDataEntry], lang="og", preprocessing=[]):  # TODO make this useless
    langgraber = get_langgraber(lang)

    data = {}
    data["id"] = [d.id for d in dataset]
    data["Text"] = [preproc.compose(*preprocessing)(*langgraber(d.get_text(lang))) for d in dataset]
    data["score"] = [d.gold_score for d in dataset]
    output = pd.DataFrame(data=data)
    return output


class BertClassifier(object):

    def __init__(self, preprocessing=[], lang="en", vocabulary=None, *, max_iter=1000):
        self.preprocessing = preprocessing
        self.model = None
        self.tokenizer = None
        self.lang = lang

    def train(self, trainingset, kfold=0, verbose=False):  # TODO generalize for use with pandas

        if kfold > 0:
            gold = []
            pred = []
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(trainingset):
                X_train = [trainingset[i] for i in train_index]
                X_test = [trainingset[i] for i in test_index]
                y_test = [trainingset[i].gold_score for i in test_index]

                self.model, self.tokenizer = train_bert.train(prepare_data(X_train, self.lang, self.preprocessing), "Text", "score")

                # TODO Test model
                predict = train_bert.predict()
                gold += y_test
                pred += list(predict)
                if verbose:
                    print(f"Fold no. {i}")
                    print(f"Accuracy = {accuracy_score(y_test, predict)}")
                    print(f"Kappa = {cohen_kappa_score(y_test, predict, weights='quadratic')}")
                i += 1
            return gold, pred
        else:
            self.model, self.tokenizer = train_bert.train(prepare_data(trainingset, self.lang, self.preprocessing), "Text", "score")
            return [], []

    def predict(self, data: CrossLingualDataEntry) -> int:  # TODO generalize for use with pandas
        # TODO support multiple predictions at one call
        return train_bert.predict()  # TODO - make the model predict something

