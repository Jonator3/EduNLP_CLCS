#!/bin/bash

echo ""
echo "=== Translate Train and Test - Logistic Regresion ==="
echo ""

echo "en > de - en > de"
sh run_logres_train_test.sh data/en_train.csv germantext data/en_test.csv germantext results/logres/Translate_Both.csv

echo "en > es - en > es"
sh run_logres_train_test.sh data/en_train.csv spanishtext data/en_test.csv spanishtext results/logres/Translate_Both.csv

echo "en300 > de (K-Fold)"
sh run_logres_kfold_subset.sh data/en_train.csv germantext results/logres/Translate_Both.csv

echo "en300 > es (K-Fold)"
sh run_logres_kfold_subset.sh data/en_train.csv spanishtext results/logres/Translate_Both.csv

echo "en_cw > de (K-Fold)"
sh run_logres_kfold.sh data/en_cw.csv germantext results/logres/Translate_Both.csv

echo "en_cw > es (K-Fold)"
sh run_logres_kfold.sh data/en_cw.csv spanishtext results/logres/Translate_Both.csv

echo "de > en (K-Fold)"
sh run_logres_kfold.sh data/de.csv englishtext results/logres/Translate_Both.csv

echo "de > es (K-Fold)"
sh run_logres_kfold.sh data/de.csv spanishtext results/logres/Translate_Both.csv

echo "es > en (K-Fold)"
sh run_logres_kfold.sh data/es.csv englishtext results/logres/Translate_Both.csv

echo "es > de (K-Fold)"
sh run_logres_kfold.sh data/es.csv germantext results/logres/Translate_Both.csv
