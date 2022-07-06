print("imports")
#import tensorflow as tf

import torch
import nltk
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences as keras_pad
#from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
#import pandas as pd
#import io
import numpy as np
import matplotlib.pyplot as plt
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
    padded = keras_pad(pad)
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


def preprocess_dataset(dataset_path, max_l=None, verbose=True):
    print("loading", dataset_path)
    data = []
    dataset = [entry for entry in main.load_data(dataset_path) if entry.set < 3 or entry.set == 10]
    if max_l is not None:
        dataset = dataset[:max_l]
    for i, D in enumerate(dataset):
        sentences = ["[CLS] " + sent + " [SEP]" for sent in nltk.tokenize.sent_tokenize(D.og_text)]
        score = D.gold_score

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_text = concat([tokenizer.tokenize(sent) for sent in sentences])
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        #input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text]

        data.append((tokenized_text, input_ids, score))
        if verbose:
            print("\r", i, "/", len(dataset), end="")
    print("\rdone\n")
    return data


print("load and tokenize data")
train_set = preprocess_dataset("data/en_train.csv", 10)
test_set = preprocess_dataset("data/en_test.csv", 10)

train_labels = [d[2] for d in train_set]
test_labels = [d[2] for d in test_set]

train_inputs = keras_pad([d[1] for d in train_set])
test_inputs = keras_pad([d[1] for d in test_set])

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

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
#model.cuda()

# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Store our loss and accuracy for plotting
train_loss_set = []
# Number of training epochs
epochs = 4

# BERT training loop
for _ in trange(epochs, desc="Epoch"):

    ## TRAINING

    # Set our model to training mode
    model.train()
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    ## VALIDATION

    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

# plot training performance

plt.figure(figsize=(15, 8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

# TODO load multilingual-BERT
