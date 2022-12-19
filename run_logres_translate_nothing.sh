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

python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_nothing.csv --k-fold 10 ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_orig en 10 --lowercase
