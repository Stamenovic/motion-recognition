# Live motion recognition pipeline

This branch prepares the post-capture part of a Vicon live recognition workflow.

## Runtime idea

1. Vicon streams segment translations continuously.
2. The server estimates the shared starting pose from recent rest frames.
3. Recording starts when hand displacement or hand speed crosses the trigger threshold.
4. The triggered segment is converted to a `TrialRecord`.
5. The same preprocessing used during training is applied.
6. The segment is classified once when movement becomes quiet or reaches max length.
7. After classification, the buffer is cleared and the server waits through cooldown.

## Current model choice

The prediction is `statistical features + linear SVM`, without temporal
normalization. The triggered segment keeps its original duration; the model uses
scalar summaries of the extracted signals instead of resampling each movement to
a fixed number of time samples.

The live server now uses automatic trigger segmentation instead of manual SPACE
segmentation. That means it waits for the shared starting pose, detects movement
start/stop, then classifies the completed segment.

The live prediction can also return `Nepoznato`. This is a reject label, not a
trained movement class. It is used when the best known-class SVM confidence is
too low or when the captured segment contains too little movement to be treated
as one of the trained actions.

## Local test without Vicon

For development outside the lab, run the live server and a fake UDP sender in
two terminals. The fake sender streams synthetic `Left`, `Right`, and `Trup`
poses in the same format expected by `scripts/vicon_live_capture.py`.

Use the Python executable from the virtual environment that exists on your
machine. The examples below use `.\.venv\Scripts\python.exe`.

### 1. Check that packets are received

Terminal A:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --probe
```

Terminal B:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py
```

The probe mode only prints incoming frames. It does not load or train a model,
so use it first to verify the port, packet format, and object names.

### 2. Run a synthetic live classification test

Terminal A:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py
```

Terminal B:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement sirenje --rest-seconds 5
```

The server does not need SPACE. It starts printing detected movements
automatically after it sees enough motion from the starting pose.
Repeat with:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement guranje --rest-seconds 5
```

The synthetic movements are only geometric approximations. They test whether
the UDP, buffering, preprocessing, and prediction path works end-to-end; they
are not a replacement for real Vicon recordings.

If the synthetic sender always predicts the same class, test with a real CSV
replay before changing the model. The synthetic sender may not match the real
movement geometry used during training.

Terminal B:

```powershell
.\.venv\Scripts\python.exe scripts\replay_csv_sender.py --list-trials
.\.venv\Scripts\python.exe scripts\replay_csv_sender.py --trial Sirenje_01 --cycles 1
```

This sends the selected CSV trial through the same UDP parser and trigger-based
prediction path as the Vicon stream.

If the same trial name appears in multiple folders, use the exact CSV path:

```powershell
.\.venv\Scripts\python.exe scripts\replay_csv_sender.py --trial-path data\raw\Sirenje_01.csv --cycles 1
```

## Train model

Install requirements first if the environment has not already been prepared:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put labelled CSV exports into `data\raw`. File names drive the labels:

```text
Sirenje_01.csv, Siri_ruke_01.csv      -> sirenje
Guranje_01.csv, Ispruzi_ruke_01.csv   -> guranje
Podizanje_desna_01.csv                 -> podizanje_desna
```

The `data\raw` folder is ignored by git, so these recordings must exist on the
machine where the live script is run.

```powershell
.\.venv\Scripts\python.exe scripts\train_live_models.py
```

This saves:

```text
models/live_motion_model.joblib
```

The file is ignored by git through the existing `models/*.joblib` rule.

For temporary live testing, `scripts\vicon_live_capture.py` now trains a fresh
model from all labelled CSV trials under `data\raw` at startup and saves it to
`models\live_motion_model.joblib`. This intentionally uses every available
recording, because the goal is to check the live path on known movement types,
not to estimate generalization accuracy.

To reuse an already saved model instead of retraining at startup:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --use-saved-model
```

## Evaluate with Leave-One-Out

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_live_models.py
```

This retrains the live-ready model on each Leave-One-Out split and reports the
statistical SVM prediction.

## Simulate a completed live segment

```powershell
.\.venv\Scripts\python.exe scripts\simulate_live_prediction.py --trial Sirenje_04
```

This loads one existing CSV trial and classifies it as if it had just been
captured from the live buffer.

## Vicon capture

`scripts/vicon_live_capture.py` is a UDP capture server. By default it binds to:

```text
host: 0.0.0.0
port: 51001
fps: 200
```

Useful options:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --host 0.0.0.0 --port 51001 --fps 200
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --baseline-frames 50
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --start-delta-mm 80
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --start-speed-mm-s 150
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --stop-speed-mm-s 200
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --stop-quiet-frames 30
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --min-frames 280
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --max-segment-frames 1000
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --cooldown-frames 100
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --probe
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --use-saved-model
```

The Vicon frame conversion must provide translations for:

```text
Left:left
Right:right
Trup:trup
```

Each triggered segment is converted to `TrialRecord` through `LiveSegmentBuffer`.

Before using the real Vicon stream, check:

- The server machine allows UDP traffic on the selected port.
- `--fps` matches the Vicon capture rate, because velocity and acceleration
  features depend on the frame duration.
- The streamed object names match `OBJECT_NAME_MAP` in
  `scripts\vicon_live_capture.py`. If probe mode prints `Left`, `Right`, and
  `Trup`, the current mapping is fine. If Nexus sends different names, change
  only the left side of that dictionary.
- If probe mode receives nothing, first check host, port, firewall, and whether
  Nexus is actually streaming.
- If probe mode receives packets but parsed frames look wrong, the packet layout
  is different from the assumed Vicon UDP Object Stream layout and the parser
  should be adjusted from a captured packet sample.

## Fake sender options

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement alternate
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement sirenje
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement guranje
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement podizanje_desna
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --format text
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --drop-rate 0.1
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --noise-mm 2
```

Important fake sender options:

```text
--host          target address; use the capture machine IP across two machines
--port          target UDP port; must match the server
--fps           stream frame rate; should match the server and training data
--movement      sirenje, guranje, podizanje_desna, or alternate
--move-seconds  duration of the synthetic movement phase
--rest-seconds  rest time between synthetic movements
--drop-rate     probability of omitting one object from a frame
--noise-mm      position noise in millimetres
```

If `vicon_live_capture.py` starts without `--use-saved-model`, a missing model
file is not an error by itself. The script will train from `data\raw`. It only
fails if no labelled CSV trials are available there.
