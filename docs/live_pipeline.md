# Live motion recognition pipeline

This branch prepares the post-capture part of a Vicon live recognition workflow.

## Runtime idea

1. Vicon streams segment translations continuously.
2. First SPACE press starts recording a movement segment.
3. Second SPACE press stops recording.
4. The buffered frames are converted to a `TrialRecord`.
5. The same preprocessing used during training is applied.
6. The saved model predicts the movement label.

## Current model choice

The prediction is `fPCA + linear SVM`.

This fits the planned SPACE start/stop workflow because the model receives a
completed movement segment, not an incomplete frame-by-frame stream.

## Local test without Vicon

For development outside the lab, run the live server and a fake UDP sender in
two terminals. The fake sender streams synthetic `Left`, `Right`, and `Trup`
poses in the same format expected by `scripts/vicon_live_capture.py`.

Use `env\Scripts\python.exe` instead of `.\.venv\Scripts\python.exe` if that is
the virtual environment name on your machine.

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

Press SPACE in the server terminal during the rest cue to start recording, wait
for the movement to finish, then press SPACE again to classify the segment.
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

This sends the selected CSV trial through the same UDP parser and SPACE
segmentation path as the Vicon stream.

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
fPCA/SVM prediction.

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
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --min-frames 10
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --probe
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --use-saved-model
```

The Vicon frame conversion must provide translations for:

```text
Left:left
Right:right
Trup:trup
```

Each recorded segment is then converted to `TrialRecord` through
`LiveSegmentBuffer`.

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
--rest-seconds  rest window for pressing SPACE
--drop-rate     probability of omitting one object from a frame
--noise-mm      position noise in millimetres
```

If `vicon_live_capture.py` starts without `--use-saved-model`, a missing model
file is not an error by itself. The script will train from `data\raw`. It only
fails if no labelled CSV trials are available there.
