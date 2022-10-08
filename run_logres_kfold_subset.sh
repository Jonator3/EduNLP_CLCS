#!/bin/bash

venv_dir=venv
source $venv_dir/bin/activate
export PYTHONPATH=$PYTHONPATH:./
python main.py --subset 300 15 --k-fold 10 $2 $1