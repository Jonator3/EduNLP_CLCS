#!/bin/bash

echo ""
echo "=== Translate Train and Test - Logistic Regresion ==="
echo ""

echo "en > de - en > de"
sh run_logres_train_test.sh de data/en_train.csv data/en_test.csv

echo "en > es - en > es"
sh run_logres_train_test.sh es data/en_train.csv data/en_test.csv

echo "en300 > de (K-Fold)"
sh run_logres_kfold_subset.sh de data/en_train.csv

echo "en300 > es (K-Fold)"
sh run_logres_kfold_subset.sh es data/en_train.csv

echo "en_cw > de (K-Fold)"
sh run_logres_kfold.sh de data/en_cw.csv

echo "en_cw > es (K-Fold)"
sh run_logres_kfold.sh es data/en_cw.csv

echo "de > en (K-Fold)"
sh run_logres_kfold.sh en data/de.csv

echo "de > es (K-Fold)"
sh run_logres_kfold.sh es data/de.csv

echo "es > en (K-Fold)"
sh run_logres_kfold.sh en data/es.csv

echo "es > de (K-Fold)"
sh run_logres_kfold.sh de data/es.csv
