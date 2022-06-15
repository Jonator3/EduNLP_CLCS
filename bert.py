print("imports")
#import tensorflow as tf

import torch
import nltk
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
#from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
#from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
#from tqdm import tqdm, trange
#import pandas as pd
#import io
#import numpy as np
import main


batch_size = 32

print("set device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
"""


def concat(list):
    out = []
    for entry in list:
        out += entry
    return out


def pad_sequences(*lists):
    pad = concat(lists)
    padded = pad_sequences(pad, maxlen=128, dtype="long", truncating="post", padding="post")
    i = 0
    out = []
    for list in lists:
        out.append(padded[i:len(list)])
        i += len(list)
    return out


def make_masks(input_ids):
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def preprocess_dataset(dataset_path, verbose=True):
    print("loading", dataset_path)
    data = []
    dataset = [entry for entry in main.load_data(dataset_path) if entry.set < 3 or entry.set == 10]
    for i, D in enumerate(dataset):
        sentences = ["[CLS] " + sent + " [SEP]" for sent in nltk.tokenize.sent_tokenize(D.og_text)]
        score = D.gold_score

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_text = concat([tokenizer.tokenize(sent) for sent in sentences])
        input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text]

        data.append((tokenized_text, input_ids, score))
        if verbose:
            print("\r", i, "/", len(dataset), end="")
    return data


print("load and tokenize data")
train_set = preprocess_dataset("data/en_train.csv")
test_set = preprocess_dataset("data/en_test.csv")

train_labels = [d[2] for d in train_set]
test_labels = [d[2] for d in test_set]

train_inputs, test_inputs = pad_sequences([d[1] for d in train_set], [d[1] for d in test_set])

train_masks = make_masks(train_inputs)
test_masks = make_masks(test_inputs)


train_inputs = torch.tensor(train_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)

# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# TODO load multilingual-BERT
# TODO train BERT
