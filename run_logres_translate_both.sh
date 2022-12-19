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

python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_en de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_en de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_en de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_en de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_en de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_en de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_en de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_en de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_en de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_en de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_en de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_en de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_en es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_en es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_en es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_en es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_en es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_en es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_en es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_en es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_en es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_en es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_en es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_en es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_en fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_en fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_en fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_en fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_en fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_en fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_en fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_en fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_en fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_en fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_en fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_en fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_en zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_en zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_en zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_en zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_en zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_en zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_en zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_en zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_en zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_en zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_en zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_en zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_en zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_de en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_de en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_de en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_de en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_de en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_de en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_de en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_de en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_de en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_de es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_de es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_de es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_de es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_de es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_de es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_de es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_de es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_de es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_de es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_de es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_de es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_de fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_de fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_de fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_de fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_de fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_de fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_de fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_de fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_de fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_de fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_de fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_de fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_de zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_de zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_de zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_de zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_de zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_de zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_de zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_de zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_de zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_de zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_de zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_de zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_de zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_es de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_es de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_es de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_es de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_es de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_es de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_es de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_es de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_es de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_es de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_es de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_es de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_es en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_es en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_es en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_es en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_es en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_es en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_es en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_es en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_es en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_es fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_es fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_es fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_es fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_es fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_es fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_es fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_es fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_es fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_es fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_es fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_es fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_es zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_es zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_es zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_es zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_es zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_es zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_es zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_es zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_es zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_es zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_es zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_es zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_es zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_fr de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_fr de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_fr de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_fr de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_fr de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_fr de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_fr de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_fr de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_fr de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_fr de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_fr de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_fr de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_fr en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_fr en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_fr en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_fr en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_fr en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_fr en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_fr en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_fr en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh en ASAP_fr en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_fr es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_fr es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_fr es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_fr es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_fr es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_fr es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_fr es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_fr es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_fr es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_fr es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_fr es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_fr es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_fr zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_fr zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_fr zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_fr zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_fr zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_fr zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_fr zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_fr zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_fr zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_fr zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_fr zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_fr zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_fr zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_zh de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_zh de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_zh de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_zh de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_zh de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_zh de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_zh de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_zh de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_zh de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_zh de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_zh de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_zh de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_zh en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_zh en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de en ASAP_zh en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_zh en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_zh en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es en ASAP_zh en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_zh en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_zh en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr en ASAP_zh en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh en 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh en 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh en 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_zh es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_zh es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_zh es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_zh es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_zh es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_zh es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_zh es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_zh es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_zh es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_zh es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_zh es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_zh es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_zh fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_zh fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_zh fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_zh fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_zh fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_zh fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_zh fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_zh fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_zh fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --k-fold 10 ASAP_zh fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_zh fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_zh fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_zh fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_orig de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_orig de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en de ASAP_orig de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_orig de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_orig de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es de ASAP_orig de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_orig de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_orig de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr de ASAP_orig de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_orig de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_orig de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh de ASAP_orig de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_orig de 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_orig de 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig de ASAP_orig de 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_orig es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_orig es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en es ASAP_orig es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_orig es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_orig es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de es ASAP_orig es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_orig es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_orig es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr es ASAP_orig es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_orig es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_orig es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh es ASAP_orig es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_orig es 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_orig es 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig es ASAP_orig es 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_orig fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_orig fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en fr ASAP_orig fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_orig fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_orig fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de fr ASAP_orig fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_orig fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_orig fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es fr ASAP_orig fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_orig fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_orig fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_zh fr ASAP_orig fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_orig fr 1 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_orig fr 2 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig fr ASAP_orig fr 10 --lowercase
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_orig zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_orig zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_en zh ASAP_orig zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_orig zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_orig zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_de zh ASAP_orig zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_orig zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_orig zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_es zh ASAP_orig zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_orig zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_orig zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_fr zh ASAP_orig zh 10 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_orig zh 1 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_orig zh 2 --classifier logres_char
python3 main.py --output ./result/logres/translate_both.csv --testset ASAP_orig zh ASAP_orig zh 10 --classifier logres_char
