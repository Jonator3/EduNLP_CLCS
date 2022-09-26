#!/bin/bash

echo ""
echo "=== Translate Test - Logistic Regresion ==="
echo ""

echo "en - de > en"
sh run_logres_train_test.sh en data/en_train.csv data/de.csv

echo "en - es > en"
sh run_logres_train_test.sh en data/en_train.csv data/es.csv

echo "en300 - de > en"
sh run_logres_train_test_subset.sh en data/en_train.csv data/de.csv

echo "en300 - es > en"
sh run_logres_train_test_subset.sh en data/en_train.csv data/es.csv

echo "en_cw - de > en"
sh run_logres_train_test.sh en data/en_cw.csv data/de.csv

echo "en_cw - es > en"
sh run_logres_train_test.sh en data/en_cw.csv data/es.csv

echo "de - en > de"
sh run_logres_train_test.sh de data/de.csv data/en_test.csv

echo "de - es > de"
sh run_logres_train_test.sh de data/de.csv data/es.csv

echo "es - en > es"
sh run_logres_train_test.sh es data/es.csv data/en_test.csv

echo "es - de > es"
sh run_logres_train_test.sh es data/es.csv data/de.csv
