import pandas
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

import preprocessing as preproc
import train_bert


class BertClassifier(object):

    def __init__(self, preprocessing=[], lang="en"):
        self.preprocessing = preprocessing
        self.model = None
        self.tokenizer = None
        self.lang = lang

    def train(self, trainingset, kfold=0, verbose=False):

        if kfold > 0:
            gold = []
            pred = []
            kf = KFold(n_splits=kfold, shuffle=True)
            i = 1
            for train_index, test_index in kf.split(trainingset):  # TODO Kfold with pandas
                X_train = [trainingset[i] for i in train_index]
                X_test = [trainingset[i] for i in test_index]
                y_test = [trainingset[i].gold_score for i in test_index]

                self.model, self.tokenizer = train_bert.train(X_train, "text", "score")

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

    def predict(self, text: str) -> int:
        return train_bert.predict()  # TODO - make the model predict something

