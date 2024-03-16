#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2


epoch=0.3
test_file="evaluation-beta-0.0-lr-1e-7-iteration-3-epoch-epoch-${epoch}-from-iteration-2-epoch-0.2-temperature-0.0.json"
helpful_win_rates_file_name="3vs1-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-3-helpful-from-iteration-2-epoch-0.2"
harmless_win_rates_file_name="3vs1-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-3-harmless-from-iteration-2-epoch-0.2"

# Run win rates calculation with Hydra configuration syntax
python win_rates.py test="$test_file" \
                    helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                    harmless_win_rates_file_name="$harmless_win_rates_file_name"
