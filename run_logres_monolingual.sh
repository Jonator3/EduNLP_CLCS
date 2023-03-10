#!/bin/bash

venv_dir=venv
if [ ! -d $venv_dir ]; then
  echo "Virtual environment not found. Setting up virtual environment under $venv_dir"
  python3 -m venv $venv_dir
  source $venv_dir/bin/activate
  pip install -r requirements.txt
  pip install -U sentence-transformers
fi
source $venv_dir/bin/activate
export PYTHONPATH=$PYTHONPATH:./


for LANG in en de es fr zh
do
  for LANG2 in en de es fr zh
  do
    if $LANG2 == zh; then
      python3 main.py --lowercase --output ./result/logres/monolingual.csv --k-fold 10 ASAP_$LANG $LANG2 "1 2 10"
    else
      python3 main.py --classifier logres_char --output ./result/logres/monolingual.csv --k-fold 10 ASAP_$LANG $LANG2 "1 2 10"
    fi
  done
done

for LANG in en de es fr zh
do
  if $LANG2 == zh; then
    python3 main.py --lowercase --output ./result/logres/monolingual.csv --testset ASAP_orig $LANG ASAP_orig $LANG "1 2 10"
  else
    python3 main.py --classifier logres_char --output ./result/logres/monolingual.csv --testset ASAP_orig $LANG ASAP_orig $LANG "1 2 10"
  fi
done

for LANG in en de es fr zh
do
  if $LANG2 == zh; then
    python3 main.py --lowercase --output ./result/logres/monolingual.csv --testset ASAP_orig300 $LANG ASAP_orig300 $LANG "1 2 10"
  else
    python3 main.py --classifier logres_char --output ./result/logres/monolingual.csv --testset ASAP_orig300 $LANG ASAP_orig300 $LANG "1 2 10"
  fi
done
