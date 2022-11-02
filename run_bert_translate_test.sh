#!/bin/bash

echo ""
echo "=== Translate Test - Bert ==="
echo ""

echo "en - de > en"
sh run_bert_train_test.sh data/en_train.csv englishtext data/de.csv englishtext results/bert/Translate_Test.csv

echo "en - es > en"
sh run_bert_train_test.sh data/en_train.csv englishtext data/es.csv englishtext results/bert/Translate_Test.csv

echo "en300 - de > en"
sh run_bert_train_test_subset.sh data/en_train.csv englishtext data/de.csv englishtext results/bert/Translate_Test.csv

echo "en300 - es > en"
sh run_bert_train_test_subset.sh data/en_train.csv englishtext data/es.csv englishtext results/bert/Translate_Test.csv

echo "en_cw - de > en"
sh run_bert_train_test.sh data/en_cw.csv englishtext data/de.csv englishtext results/bert/Translate_Test.csv

echo "en_cw - es > en"
sh run_bert_train_test.sh data/en_cw.csv englishtext data/es.csv englishtext results/bert/Translate_Test.csv

echo "de - en > de"
sh run_bert_train_test.sh data/de.csv germantext data/en_test.csv germantext results/bert/Translate_Test.csv

echo "de - es > de"
sh run_bert_train_test.sh data/de.csv germantext data/es.csv germantext results/bert/Translate_Test.csv

echo "es - en > es"
sh run_bert_train_test.sh data/es.csv spanishtext data/en_test.csv spanishtext results/bert/Translate_Test.csv

echo "es - de > es"
sh run_bert_train_test.sh data/es.csv spanishtext data/de.csv spanishtext results/bert/Translate_Test.csv
