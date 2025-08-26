#!/bin/bash

if [ -d "workspace" ]; then
  echo "Workspace already exists. Please remove it before retraining."
  exit 1
fi

source tf_modelbuildenv

python src/setup_tf_retrain_env.py --rootpath . --datapath tf_dataset
python src/retrain_tf_model.py