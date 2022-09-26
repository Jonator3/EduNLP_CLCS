#!/bin/bash

echo ""
echo "=== Monolingual - Logistic Regresion ==="
echo ""

echo "en - en"
sh run_logres_train_test.sh og data/en_train.csv data/en_test.csv

echo "en300 (K-Fold)"
sh run_logres_kfold_subset.sh og data/en_train.csv

echo "en_cw (K-Fold)"
sh run_logres_kfold.sh og data/en_cw.csv

echo "de (K-Fold)"
sh run_logres_kfold.sh og data/de.csv

echo "es (K-Fold)"
sh run_logres_kfold.sh og data/es.csv
