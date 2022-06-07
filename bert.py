# verify GPU availability
import time

import tensorflow as tf

import torch
import nltk
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
#import pandas as pd
import io
import numpy as np

import main


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
"""

data = []
for D in main.load_data("data/de.csv"):
    sentences = [["[CLS]"]+nltk.tokenize.word_tokenize(sent)+["[SEP]"] for sent in nltk.tokenize.sent_tokenize(D.og_text)]
    score = D.gold_score

    # TODO BERT-Tokenize
    data.append((sentences, score))

# TODO load multilingual-BERT
# TODO train BERT
