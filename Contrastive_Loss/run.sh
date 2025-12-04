#!/bin/bash

# Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
source C:/Users/tanhs/miniconda3/etc/profile.d/conda.sh
conda activate mil

python main.py --config config/camelyon16_medical_ssl_config.yml
python main.py --config config/camelyon17_medical_ssl_config.yml
python main.py --config config/bracs_medical_ssl_config.yml
