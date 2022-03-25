

class CrossLingualContendScoring(object):

    def __init__(self, trainingset, vocabulary=None, preprocessing=[]):
        self.preprocessing = preprocessing
        self.vocab = vocabulary
        if vocabulary is None:
            self.vocab = get_vocabulary(trainingset)
        count_matrix = self.create_features(trainingset)
        self.svc = svm.SVC()
        self.svc.fit(count_matrix, [data_entry.gold_score for data_entry in trainingset])

    def create_features(self, data: List[CrossLingualDataEntry]):
        count_matrix = self.vocab.transform([preprocessing.compose(*self.preprocessing)(data_entry.en_text) for data_entry in data])
        return count_matrix

    def predict(self, data: CrossLingualDataEntry) -> int:
        return self.svc.predict(self.create_features([data]))[0]


def load_data(input_path: str) -> List[CrossLingualDataEntry]:
    data = []
    reader = csv.reader(open(input_path, "r"))
    reader.__next__()  # skip head
    for row in reader:
        entry = CrossLingualDataEntry(row[0], row[4], row[1], row[2], row[3], row[5])
        data.append(entry)
    return data


def validate(svm: CrossLingualContendScoring, dataset: List[CrossLingualDataEntry]):
    mat = []
    for i1 in range(4):
        row = []
        for i2 in range(4):
            row.append(0)
        mat.append(row)
    for data in dataset:
        predict = svm.predict(data)
        gold = data.gold_score
        mat[gold][predict] += 1
    return mat


def get_vocabulary(*datasets):
    set = []
    for d in datasets:
        set.extend(d.copy())

    vocab = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    vocab.fit([data.en_text for data in set])
    return vocab


def print_validation(mat):
    print("Gold/\tPrediction")
    print("\t|", "0\t\t|", "1\t\t|", "2\t\t|", "3")
    for g, pre in enumerate(mat):
        print(g, "\t|", pre[0], "\t|", pre[1], "\t|", pre[2], "\t|", pre[3])


if __name__ == "__main__":
    en_train = load_data("data/en_train.csv")
    en_test = load_data("data/en_test.csv")
    de_test = load_data("data/de.csv")
    es_test = load_data("data/es.csv")

    vocabulary = get_vocabulary(en_test, en_train, de_test, es_test)
    preproc = [preprocessing.lower]
    svm = CrossLingualContendScoring(en_train, vocabulary, preproc)
    pickle.dump(svm, open("en_only.clcs", "wb"))  # save the svm to file
    print("Training Done!")

    print("English:")
    en_val = validate(svm, en_test)
    print_validation(en_val)
    print("")
    print("Spanish:")
    es_val = validate(svm, es_test)
    print_validation(es_val)
    print("")
    print("German:")
    de_val = validate(svm, de_test)
    print_validation(de_val)
    print("")


