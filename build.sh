#!/usr/bin/env bash
set -e

pyinstaller --onefile \
  --add-data "epoch-18_valAcc-0.735.h5:." \
  --add-data "sign_to_prediction_index_map.json:." \
  --collect-data mediapipe \
  model_test.py
