#!/usr/bin/env bash

# versions list: https://repo.anaconda.com/archive/
# should be a Python3.7 version thanks to `apsw`
ANACONDA=Anaconda3-2020.02-Linux-x86_64.sh

echo "Getting Anaconda3-2011.11..."
wget https://repo.anaconda.com/archive/$ANACONDA

echo "Running installation; do whatever the instruction says."
bash ./$ANACONDA

rm -rf ./$ANACONDA

echo "Now you need to close and re-open the terminal to enable working via Anaconda."
