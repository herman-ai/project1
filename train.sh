#! /bin/bash

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

rm -rf experiments/reference/c*

