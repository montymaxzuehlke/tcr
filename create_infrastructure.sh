#!/bin/bash 

mkdir automl
mkdir H2O_Processing
mkdir Results
mkdir Results_REF
mkdir Processing
mkdir Processing_DA
mkdir r_Processing_DA

echo "Built Infrastructure Part 1"

python3 Utils/infrastructure.py

echo "Built Infrastructure Part 2"

exit 0
