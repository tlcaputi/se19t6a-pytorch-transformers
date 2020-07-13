#!/bin/bash

# Setup virtual environment
pyenv virtualenv 3.6.10 se19t6a-pytorch-transformers
pyenv local se19t6a-pytorch-transformers

# Clean the directory
rm -rf data/ || echo "no data directory to delete"
rm -rf output/ || echo "no output directory to delete"

# Install packages
pip install -r requirements.txt

# Download data
mkdir data
wget https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip
unzip OLIDv1.0.zip -d data
rm OLIDv1.0.zip

# Run code
python code/se19.py --ROOTPATH $(pwd)
