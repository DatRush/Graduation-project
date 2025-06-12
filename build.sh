#!/usr/bin/env bash

pyinstaller --onefile \
  --add-data "epoch-18_valAcc-0.735.h5:." \
  --add-data "sign_to_prediction_index_map.json:." \
  model_test.py
