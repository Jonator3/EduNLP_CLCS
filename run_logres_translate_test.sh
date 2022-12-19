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

python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_en en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_en en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig en ASAP_en en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en de ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en de ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en de ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es de ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es de ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es de ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr de ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr de ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr de ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh de ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh de ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh de ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig de ASAP_de de 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig de ASAP_de de 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig de ASAP_de de 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en es ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en es ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en es ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de es ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de es ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de es ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr es ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr es ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr es ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh es ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh es ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh es ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig es ASAP_es es 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig es ASAP_es es 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig es ASAP_es es 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en fr ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en fr ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en fr ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de fr ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de fr ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de fr ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es fr ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es fr ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es fr ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh fr ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh fr ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh fr ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig fr ASAP_fr fr 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig fr ASAP_fr fr 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig fr ASAP_fr fr 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en zh ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en zh ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en zh ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de zh ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de zh ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de zh ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es zh ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es zh ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es zh ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr zh ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr zh ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr zh ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig zh ASAP_zh zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig zh ASAP_zh zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_orig zh ASAP_zh zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_en en ASAP_orig en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_de en ASAP_orig en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_es en ASAP_orig en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_fr en ASAP_orig en 10 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_orig en 1 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_orig en 2 --lowercase
python3 main.py --output ./result/logres/translate_test.csv --testset ASAP_zh en ASAP_orig en 10 --lowercase
