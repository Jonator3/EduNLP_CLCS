#!/bin/bash

echo ""
echo "=== Translate Train - Logistic Regresion ==="
echo ""

echo "en > de - de"
sh run_logres_train_test.sh de data/en_train.csv data/de.csv

echo "en > es - es"
sh run_logres_train_test.sh es data/en_train.csv data/es.csv

echo "en300 > de - de"
sh run_logres_train_test_subset.sh de data/en_train.csv data/de.csv

echo "en300 > es - es"
sh run_logres_train_test_subset.sh es data/en_train.csv data/es.csv

echo "en_cw > de - de"
sh run_logres_train_test.sh de data/en_cw.csv data/de.csv

echo "en > es - es"
sh run_logres_train_test.sh es data/en_cw.csv data/es.csv

echo "de > en - en"
sh run_logres_train_test.sh en data/de.csv data/en_test.csv

echo "de > es - es"
sh run_logres_train_test.sh es data/de.csv data/es.csv

echo "es > en - en"
sh run_logres_train_test.sh en data/es.csv data/en_test.csv

echo "es > de - de"
sh run_logres_train_test.sh de data/es.csv data/de.csv

