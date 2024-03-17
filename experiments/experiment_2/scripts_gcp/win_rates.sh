#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2
echo "SEED"
epoch=0.3
test_file="evaluation-beta-0.0-lr-1e-7-iteration-3-epoch-epoch-${epoch}-temperature-0.0.json"
helpful_win_rates_file_name="3vs0-epoch-${epoch}-helpful_seed_1"
harmless_win_rates_file_name="3vs0-epoch-${epoch}-harmless_seed_1"

python win_rates.py test="$test_file" \
                    helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                    harmless_win_rates_file_name="$harmless_win_rates_file_name"
