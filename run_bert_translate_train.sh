#!/bin/bash

echo ""
echo "=== Translate Train - Bert ==="
echo ""

echo "en > de - de"
sh run_bert_train_test.sh data/en_train.csv germantext data/de.csv germantext results/bert/Translate_Train.csv

echo "en > es - es"
sh run_bert_train_test.sh data/en_train.csv spanishtext data/es.csv spanishtext results/bert/Translate_Train.csv

echo "en300 > de - de"
sh run_bert_train_test_subset.sh data/en_train.csv germantext data/de.csv germantext results/bert/Translate_Train.csv

echo "en300 > es - es"
sh run_bert_train_test_subset.sh data/en_train.csv spanischtext data/es.csv spanischtext results/bert/Translate_Train.csv

echo "en_cw > de - de"
sh run_bert_train_test.sh data/en_cw.csv germantext data/de.csv germantext results/bert/Translate_Train.csv

echo "en > es - es"
sh run_bert_train_test.sh data/en_cw.csv spanishtext data/es.csv spanishtext results/bert/Translate_Train.csv

echo "de > en - en"
sh run_bert_train_test.sh data/de.csv englishtext data/en_test.csv englishtext results/bert/Translate_Train.csv

echo "de > es - es"
sh run_bert_train_test.sh data/de.csv spanishtext data/es.csv spanishtext results/bert/Translate_Train.csv

echo "es > en - en"
sh run_bert_train_test.sh data/es.csv englishtext data/en_test.csv englishtext results/bert/Translate_Train.csv

echo "es > de - de"
sh run_bert_train_test.sh data/es.csv germantext data/de.csv germantext results/bert/Translate_Train.csv

