#! /bin/bash


if [ -z "$1" ]
then
      echo "Nothing specified to run"
else
  echo running [${1}]
  python experiments/model_main_tf2.py --model_dir=experiments/${1}/ --pipeline_config_path=experiments/${1}/pipeline_new.config

  rm -rf experiments/${1}/c*
fi
