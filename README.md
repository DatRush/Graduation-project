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
python model_test.py [--camera N]
```

Use `--camera` to specify the camera index if your system has multiple cameras. For example, `--camera 1` may select the built‑in webcam on macOS when an iPhone is connected via Continuity Camera.

A window will open showing the webcam feed with predicted labels displayed at the bottom of the frame.

## Building a standalone app

To package the demo as a single executable so it can run without Python installed, first install **PyInstaller**:

```bash
pip install pyinstaller
```

Run the helper script to build the app:

On macOS or Linux:
```bash
./build.sh
```
On Windows use the provided batch file (or run `bash build.sh` from Git Bash):
```cmd
build.bat
```

Make sure to run these commands from a terminal window rather than double-
clicking the script file.

The script uses `set -e` so the build will stop if PyInstaller encounters an
error.

The executable will appear in the `dist/` folder. Launch it with an optional camera index just like the script:

```bash
./dist/model_test [--camera N]
```

If you make changes to the Python code or model files, simply run `./build.sh` again to produce an updated executable.
