#!/bin/bash

python3 -m venv venv/
# assumming that you're using zsh
source venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install numpy scipy matplotlib
