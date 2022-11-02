#!/bin/bash

echo ""
echo "=== Translate Train and Test - Bert ==="
echo ""

echo "en > de - en > de"
sh run_bert_train_test.sh data/en_train.csv germantext data/en_test.csv germantext results/bert/Translate_Both.csv

echo "en > es - en > es"
sh run_bert_train_test.sh data/en_train.csv spanishtext data/en_test.csv spanishtext results/bert/Translate_Both.csv

echo "en300 > de (K-Fold)"
sh run_bert_kfold_subset.sh data/en_train.csv germantext results/bert/Translate_Both.csv

echo "en300 > es (K-Fold)"
sh run_bert_kfold_subset.sh data/en_train.csv spanishtext results/bert/Translate_Both.csv

echo "en_cw > de (K-Fold)"
sh run_bert_kfold.sh data/en_cw.csv germantext results/bert/Translate_Both.csv

echo "en_cw > es (K-Fold)"
sh run_bert_kfold.sh data/en_cw.csv spanishtext results/bert/Translate_Both.csv

echo "de > en (K-Fold)"
sh run_bert_kfold.sh data/de.csv englishtext results/bert/Translate_Both.csv

echo "de > es (K-Fold)"
sh run_bert_kfold.sh data/de.csv spanishtext results/bert/Translate_Both.csv

echo "es > en (K-Fold)"
sh run_bert_kfold.sh data/es.csv englishtext results/bert/Translate_Both.csv

echo "es > de (K-Fold)"
sh run_bert_kfold.sh data/es.csv germantext results/bert/Translate_Both.csv
