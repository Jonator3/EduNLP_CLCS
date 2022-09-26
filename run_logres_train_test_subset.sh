#!/bin/bash

venv_dir=venv
source $venv_dir/bin/activate
export PYTHONPATH=$PYTHONPATH:./
python main.py --subset 300 15 --testset $3 $2 $1
