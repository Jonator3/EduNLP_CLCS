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

echo ""
echo "=== Test Run - Bert ==="
echo ""

echo "Test subset"
echo "en-16 (K-Fold)"
python main.py --classifier bert --bert_batch_size 8 --output results/bert/Test.csv --subset 16 2 --k-fold 5 data/en_train.csv --trainset_text originaltext


