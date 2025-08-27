#!/bin/bash

if [ -d "workspace" ]; then
  echo "Workspace already exists."
  read -p "Do you want to continue anyway? (y/N): " choice
  case "$choice" in 
    y|Y ) echo "Continuing despite existing workspace...";;
    * ) echo "Exiting."; exit 1;;
  esac
fi

source tf_modelbuildenv

python src/setup_tf_retrain_env.py --rootpath . --datapath tf_dataset
python src/retrain_tf_model.py
