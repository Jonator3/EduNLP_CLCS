import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import softmax
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import datetime


def lr_schedule(epoch, epochs):
    peak = epochs * 0.1
    if epoch <= peak:
        return epoch / peak
    else:
        return 1 - ((epoch - peak) / (epochs - peak))


def to_tensor_dataset(data_frame, index, input_column, target_column, label2id, tokenizer):
    indices = data_frame.index[index]
    sentences = data_frame.loc[indices, input_column].fillna('').to_list()
    labels = data_frame.loc[indices, target_column].map(label2id).to_list()

    inputs = tokenizer(sentences, return_tensors='pt', max_length=None, padding='max_length', truncation=True)

    return TensorDataset(inputs['input_ids'],
                         inputs['attention_mask'],
                         torch.tensor(labels),
                         torch.tensor(indices))


def generate_train_test_dataset(data_frame, train_idxs, test_idxs, input_col, target_col, label2id, tokenizer):
    train_set = to_tensor_dataset(data_frame, train_idxs, input_col, target_col, label2id, tokenizer)
    test_set = to_tensor_dataset(data_frame, test_idxs, input_col, target_col, label2id, tokenizer)
    return train_set, test_set


def load_bert_model(bert, device, labels, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(bert)
    model = AutoModelForSequenceClassification.from_pretrained(bert).to(device)

    if labels.size > 2:
        model.config.num_labels = labels.size
        model.classifier = Linear(in_features=model.config.hidden_size,
                                  out_features=labels.size,
                                  bias=True).to(device)

    return model, tokenizer


def predict(model, tokenizer):
    pass  # TODO make the model predict something


def calculate_loss(model, tokens, mask, targets, loss_fn):
    return loss_fn(model(tokens, mask)[0], targets)


def predict_labels(model, tokens, mask):
    _, pred_labels = torch.max(softmax(model(tokens, mask)[0], 1), 1)
    return pred_labels


def compute_class_weights(label2id, y):
    n_samples = len(y)
    n_classes = len(label2id)
    return n_samples / (n_classes * (np.bincount(pd.Series(y).map(label2id)) + 1))


def train(df_train, input_col, target_cols):

    np.random.seed(4669)
    epochs = 6
    batch_size = 16
    loss_type = CrossEntropyLoss
    optimizer = AdamW
    learning_rate = 2e-5
    weight_decay = 0.01
    n_folds = 10

    bert = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4669)

    for target_col in target_cols:
        labels = df_train[target_col].unique()
        labels.sort()
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        weigh_classes = np.all(df_train[target_col].value_counts() > 0)

        df = df_train
        splits = cv.split(df[input_col], df[target_col])
        start_fold = 1

        for i, idxs in enumerate(splits, start=start_fold):
            train_idx, test_idx = idxs

            if weigh_classes:
                y = df.loc[df.index[train_idx], target_col].values
                class_weights = compute_class_weights(label2id, y).astype(np.float32)
                class_weights = torch.from_numpy(class_weights).to(device)
                loss_function = loss_type(class_weights)
            else:
                loss_function = loss_type()

            model, tokenizer = load_bert_model(bert, device, labels, id2label, label2id)
            opt = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            train_set, test_set = generate_train_test_dataset(df, train_idx, test_idx,
                                                              input_col, target_col,
                                                              label2id, tokenizer)

            model.train()
            for epoch in range(epochs):
                print(datetime.datetime.now(), "Starting epoch", epoch+1, "out of", epochs)
                for j, batch in enumerate(DataLoader(train_set, batch_size=batch_size, shuffle=True)):
                    tokens, mask, targets, idx = (x.to(model.device) for x in batch)
                    opt.zero_grad()
                    loss = calculate_loss(model, tokens, mask, targets, loss_function)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                y_true = []
                y_predicted = []

                for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False):
                    tokens, mask, targets, idx = (x.to(model.device) for x in batch)
                    pred_labels = predict_labels(model, tokens, mask)

                    y_true.extend(targets.cpu().tolist())
                    y_predicted.extend(pred_labels.cpu().tolist())

            return model, tokenizer
