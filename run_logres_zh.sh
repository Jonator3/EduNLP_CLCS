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
export PYTHONPATH=$PYTHONPATH:./python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/de/prompt_1.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/de/prompt_2.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/de/prompt_10.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/en/prompt_1.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/en/prompt_2.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/en/prompt_10.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/es/prompt_1.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/es/prompt_2.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/es/prompt_10.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/fr/prompt_1.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/fr/prompt_2.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/fr/prompt_10.tsv --lowercase
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/zh/prompt_1.tsv --classifier logres_char
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/zh/prompt_2.tsv --classifier logres_char
python main.py --output ./result/logres/zh.csv --k-fold 10 data/zh/zh/prompt_10.tsv --classifier logres_char
