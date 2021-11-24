#!/bin/bash
mv raw/person/* raw
mv raw/nothing/* raw
python convert.py
time python train.py
python infer.py > cp-to-categories.sh
./cp-to-categories.sh
