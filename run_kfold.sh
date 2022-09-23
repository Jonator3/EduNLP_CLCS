#!/bin/bash

venv_dir=venv
source $venv_dir/bin/activate
export PYTHONPATH=$PYTHONPATH:./
python main.py --k-fold 10 $2 $1