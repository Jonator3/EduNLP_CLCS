#!/bin/bash

echo ""
echo "=== Test Run - Bert ==="
echo ""


echo "Test normal"
echo "en_cw - en"
sh run_bert_train_test.sh data/en_cw.csv originaltext data/en_test.csv originaltext results/bert/Test.csv

echo "Test KFold"
echo "de (K-Fold)"
sh run_bert_kfold.sh data/de.csv originaltext results/bert/Test.csv

echo "Test subset"
echo "en300 (K-Fold)"
sh run_bert_kfold_subset_minimal.sh data/en_train.csv originaltext results/bert/Monolingual.csv

