#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2


epoch=1.0
test_file="evaluation-beta-0.0-lr-1e-7-iteration-1-epoch-epoch-${epoch}-temperature-0.0.json"
helpful_win_rates_file_name="1vs0-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-1-helpful"
harmless_win_rates_file_name="1vs0-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-1-harmless"

# Run win rates calculation with Hydra configuration syntax
python win_rates.py test="$test_file" \
                    helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                    harmless_win_rates_file_name="$harmless_win_rates_file_name"
