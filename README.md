# ASL Recognition Demo

This repository contains a simple demonstration of a model for recognizing American Sign Language (ASL) gestures. The script captures frames from a webcam and prints the most probable sign on top of the video.

## Setup

1. Install Python (version 3.10 or newer is recommended).
2. Optionally create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Place the provided model weights `epoch-18_valAcc-0.735.h5` and the mapping file `sign_to_prediction_index_map.json` in the same folder as `model_test.py` (they are already included in this repository).

## Running the demo

Make sure a webcam is connected. To start the real-time recognition demo, run:

```bash
python model_test.py
```

A window will open showing the webcam feed with predicted labels displayed at the bottom of the frame.
