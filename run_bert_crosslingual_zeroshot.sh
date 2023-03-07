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
    python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_$LANG $LANG ASAP_$LANG2 $LANG2 "1 2 10"
  done
done

for LANG in en de es fr zh
do
  python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_$LANG $LANG "1 2 10"
  python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_$LANG $LANG ASAP_orig en "1 2 10"
done

for LANG in en de es fr zh
do
  python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig300 en ASAP_$LANG $LANG "1 2 10"
  python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_$LANG $LANG ASAP_orig300 en "1 2 10"
done


