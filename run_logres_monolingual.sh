#!/bin/bash

echo ""
echo "=== Monolingual - Logistic Regresion ==="
echo ""

echo "en - en"
sh run_logres_train_test.sh data/en_train.csv originaltext data/en_test.csv originaltext results/logres/Monolingual.csv

echo "en300 (K-Fold)"
sh run_logres_kfold_subset.sh data/en_train.csv originaltext results/logres/Monolingual.csv

echo "en_cw (K-Fold)"
sh run_logres_kfold.sh data/en_cw.csv originaltext results/logres/Monolingual.csv

echo "de (K-Fold)"
sh run_logres_kfold.sh data/de.csv originaltext results/logres/Monolingual.csv

echo "es (K-Fold)"
sh run_logres_kfold.sh data/es.csv originaltext results/logres/Monolingual.csv
