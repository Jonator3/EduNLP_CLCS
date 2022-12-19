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

python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_en en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_en en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_en en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_en en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_en en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_en en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_en en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_en en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_en en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_en en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_en en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_en en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_en en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_en en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_en en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_de de 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_de de 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_de de 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_de de 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_de de 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_de de 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_de de 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_de de 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_de de 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_de de 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_de de 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_de de 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_de de 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_de de 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_de de 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_es es 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_es es 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_es es 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_es es 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_es es 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_es es 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_es es 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_es es 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_es es 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_es es 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_es es 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_es es 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_es es 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_es es 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_es es 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_fr fr 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_fr fr 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_fr fr 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_fr fr 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_fr fr 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_fr fr 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_fr fr 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_fr fr 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_fr fr 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_fr fr 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_fr fr 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_fr fr 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_fr fr 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_fr fr 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_fr fr 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_zh zh 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_zh zh 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_zh zh 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_zh zh 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_zh zh 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_zh zh 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_zh zh 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_zh zh 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_zh zh 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_zh zh 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_zh zh 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_zh zh 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_zh zh 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_zh zh 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_orig en ASAP_zh zh 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_orig en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_orig en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_en en ASAP_orig en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_orig en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_orig en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_de de ASAP_orig en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_orig en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_orig en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_es es ASAP_orig en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_orig en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_orig en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_fr fr ASAP_orig en 10
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_orig en 1
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_orig en 2
python3 main.py --classifier bert --output ./result/bert/crosslingual.csv --testset ASAP_zh zh ASAP_orig en 10
