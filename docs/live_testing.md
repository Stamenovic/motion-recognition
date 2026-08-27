# Testing the live pipeline without Vicon

`scripts/fake_vicon_sender.py` streams synthetic Left/Right/Trup poses over UDP
in the same format `scripts/vicon_live_capture.py` expects. That lets you test
the whole live path on one machine, with no Vicon and no lab.

The sender uses only the Python standard library, so it runs even before the
project requirements are installed. The capture server does not: it imports
`src.feature_extraction`, `src.live_capture` and `src.live_model` at module
level, so it needs numpy, scikit-learn, scikit-fda and joblib in **every** mode,
including `--probe`.

> **Note on the interpreter path.** The commands below use `env\Scripts\python.exe`
> because that was the original documented environment name. In the current
> checkout we have been running `.\.venv\Scripts\python.exe`; use whichever one
> exists on your machine.

---

## Step 1 - install the requirements

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This is required before the capture server will start in any mode.

---

## Step 2 - check the stream

Open **two terminals** in the project root.

Terminal A, the receiver in probe mode:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --probe
```

Terminal B, the sender:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py
```

Terminal A should print frames like:

```text
frame 41:
    Left:left: (-250.2, 149.5, 1249.9)
    Right:right: (250.6, 150.2, 1250.5)
    Trup:trup: (-0.1, 0.3, 999.9)
```

If you see this, the socket, the packet format and the object-name mapping all
work. Stop both with Ctrl+C.

Probe mode does not load the model bundle, so this step works before you have
a trained model. It still needs the requirements from Step 1.

At 200 fps probe prints roughly 800 lines per second. Let it run for a second,
then Ctrl+C and scroll back - you only need to read one frame block.

---

## Step 3 - put data in place

### Where the files go

Export the CSVs **flat** into `data\raw\`:

```text
data\raw\
    Sirenje_01.csv
    Sirenje_02.csv
    Guranje_01.csv
    Guranje_02.csv
```

Do **not** create folders named after the labels. The label is read from the
file name, never from the folder.

Sub-folders are allowed but cosmetic. `parse_trial_path` fills `patient` and
`session` only when a file sits exactly two levels deep
(`data\raw\Pacijent1\Sesija1\Sirenje_01.csv`), and neither value is used in
training or prediction. A single folder level gives you nothing.

### How the label is decided

The file stem is lowercased and common separators are ignored. That means
underscore, hyphen or no separator before the number are accepted.

| Prefix | Label |
| --- | --- |
| `sirenje`, `siri_ruke` | `sirenje` |
| `guranje`, `ispruzi_ruke` | `guranje` |
| `podizanje_desna` | `podizanje_desna` |

| Example | Result |
| --- | --- |
| `Sirenje_01.csv` | accepted, label `sirenje` |
| `sirenje_01.csv`, `SIRENJE_01.csv` | accepted, matching is case-insensitive |
| `Sirenje01.csv`, `Sirenje-01.csv` | accepted, label `sirenje` |
| `Siri_ruke_01.csv` | accepted, label `sirenje` |
| `Podizanje_desna01.csv`, `Podizanje_desna_01.csv` | accepted, label `podizanje_desna` |
| `Širenje_01.csv` | **rejected** - the alias table is plain ASCII `sirenje` |
| `Trial_Sirenje_01.csv` | rejected - the name must *start* with the prefix |

Files that do not match are skipped **silently**, with no warning. If only some
files are misnamed you will train on fewer trials than you expect.

### Recommended names

```text
Sirenje_01.csv  Sirenje_02.csv  Sirenje_03.csv  Sirenje_04.csv
Guranje_01.csv  Guranje_02.csv  Guranje_03.csv  Guranje_04.csv
Podizanje_desna_01.csv  Podizanje_desna_02.csv
```

Labelling itself is case-insensitive, but `scripts/simulate_live_prediction.py`
and `scripts/generate_review_diagnostics.py` look trials up by stem
**case-sensitively** and expect exactly these names.

### Verify before training

This check needs no installed packages - `src/path_parser.py` imports only
`pathlib`:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from pathlib import Path; from src.path_parser import parse_label_from_name as f; ps=sorted(Path('data/raw').rglob('*.csv')); print(f'{len(ps)} csv files'); [print(('  OK   ' if f(p.name) else '  SKIP '), p.name, '->', f(p.name)) for p in ps]"
```

Every file must report `OK`. Any `SKIP` will be ignored by training.

---

## Step 4 - train

```powershell
.\.venv\Scripts\python.exe scripts\train_live_models.py
```

This writes `models\live_motion_model.joblib`. The capture server also trains a
fresh model from `data\raw` at startup by default; this manual step is useful
when you want to refresh and inspect the model before live capture.

On `feat/live2`, this live model does not time-normalize each trial. It keeps
the original sample spacing and pads shorter completed segments with their
final value so all fPCA rows have the same length. If you do not pass
`--fixed-num-samples`, training uses the longest valid recording as that fixed
length.

Check the printed `Loaded trials:` count against the number of CSVs you expect.
A lower number means some files were skipped by the naming rule above.

---

## Step 5 - run the live test

Arrange the two terminals **side by side**. The capture server now detects
movement start/stop automatically; it does not need SPACE to mark movement
start and end.

Terminal A, the capture server:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py
```

Terminal B, the sender:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --movement sirenje
```

The sender counts down to each movement:

```text
[001] rest 2.0s
        MOVE in 2s
        MOVE in 1s
[001] MOVE sirenje (3.0s)
[001] done
```

Terminal A estimates the shared starting pose from 50 rest frames. It starts
recording when hand displacement passes 80 mm or hand speed passes 150 mm/s.
It stops when hand speed stays below 200 mm/s for 30 frames, or when the segment
reaches 1000 frames. Segments shorter than 280 frames are not classified. After
classification it clears the buffer and waits through a 100-frame cooldown.
Padding is applied only after this stop condition closes the segment; the
active stream is not padded frame by frame.

Terminal A prints:

```text
Baseline frames: 50
Start delta: 80.0 mm
Start speed: 150.0 mm/s
Stop speed: 200.0 mm/s
Stop quiet frames: 30
Minimum segment frames: 280
Maximum segment frames: 1000
Cooldown after segment: 100 complete frame(s)
Detected movement 1: label=sirenje, frames=335, range=704-1039, candidate=sirenje, confidence=2.247, motion=813.1 mm, reason=quiet
```

If the triggered segment does not look confident enough, or if there is too
little movement in it, the final prediction becomes:

```text
Detected movement 2: label=nepoznato, frames=280, range=3400-3679, candidate=guranje, confidence=2.225, motion=0.0 mm, reason=quiet
```

`nepoznato` is not trained as a fourth movement class. It is a reject label used
when the model should avoid forcing the segment into `guranje`,
`podizanje_desna`, or `sirenje`.

Press **Q** or Ctrl+C in terminal A to quit.

Repeat with `--movement guranje` to check the other class.

---

## Command reference

### `scripts/fake_vicon_sender.py`

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Where to send. Use the capture machine's IP for two machines. |
| `--port` | `51001` | Target UDP port. Must match the server. |
| `--fps` | `200` | Frame rate. Must match the server and the training CSVs. |
| `--movement` | `alternate` | `sirenje`, `guranje`, `podizanje_desna`, or `alternate` between them. |
| `--move-seconds` | `3.0` | Duration of the movement. Recorded trials run 1.3-5.4 s. |
| `--rest-seconds` | `2.0` | Rest between movements, with a per-second countdown. |
| `--cycles` | `0` | Stop after N movements. `0` runs until Ctrl+C. |
| `--format` | `binary` | `binary` (Vicon Object Stream) or `text` fallback. |
| `--drop-rate` | `0.0` | Probability of dropping one object per frame, to fake occlusion. |
| `--noise-mm` | `0.5` | Position noise, so the Butterworth filter has something to do. |

The synthetic movements are geometric approximations, but the geometry is
tuned to the recorded CSVs in `data\raw`. Hand separation is 500 mm at rest,
ramps to about 1490 mm for `sirenje`, and closes to about 100 mm for `guranje`
while both hands travel forward. `podizanje_desna` raises the right hand. That
puts the mean separation inside the
recorded bands (guranje 150-377 mm, sirenje 1147-1318 mm), so a tightly
captured segment should classify correctly.

The movements ramp to full amplitude and hold rather than returning to rest,
because the recorded trials end in the extended pose.

### `scripts/vicon_live_capture.py`

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Local address to bind. `0.0.0.0` accepts from any machine. |
| `--port` | `51001` | UDP port to bind. |
| `--fps` | `200` | Capture rate. **Must match the training CSVs**, which are 200 Hz. |
| `--model-path` | `models/live_motion_model.joblib` | Where the trained model bundle is saved or loaded from. |
| `--use-saved-model` | off | Load `--model-path` instead of retraining from `data\raw` at startup. |
| `--baseline-frames` | `50` | Rest frames used to estimate the starting pose. |
| `--start-delta-mm` | `80` | Start recording when hand displacement from baseline exceeds this. |
| `--start-speed-mm-s` | `150` | Start recording when hand speed exceeds this. |
| `--stop-speed-mm-s` | `200` | Count a recording frame as quiet when hand speed is below this. |
| `--stop-quiet-frames` | `30` | Stop recording after this many quiet frames. |
| `--min-frames` | `280` | Do not classify triggered segments shorter than this. |
| `--max-segment-frames` | `1000` | Force classification when a segment reaches this length. |
| `--fixed-num-samples` | longest training recording | fPCA length used for padding/truncation when training at startup. |
| `--cooldown-frames` | `100` | Ignore this many complete frames after a segment is classified. |
| `--probe` | off | Print parsed frames, do not load the model. |

Keys: **Q** or Ctrl+C to quit. SPACE is ignored in trigger mode.

The live model reports both the final label and the best known-class candidate.
For example, `fPCA=nepoznato, candidate=guranje` means the SVM's closest known
class was `guranje`, but the confidence or movement amount was below the
accepted threshold.

---

## Testing across two machines

Run the sender on one machine and the server on the other:

```powershell
.\.venv\Scripts\python.exe scripts\fake_vicon_sender.py --host 192.168.1.50
```

The server binds `0.0.0.0` by default, so it accepts remote packets. Allow
UDP 51001 through Windows Firewall on the receiving machine, or add an
inbound rule for `python.exe`.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Probe prints nothing | Wrong port, or firewall | Match `--port` on both sides; allow UDP through the firewall |
| Probe prints `Left`, not `Left:left` | Object names not mapped | Edit `OBJECT_NAME_MAP` at the top of `vicon_live_capture.py` |
| `ModuleNotFoundError: numpy` or `joblib` | Requirements not installed | Run Step 1 - this affects `--probe` too |
| `No labeled CSV trials found` | Every file failed the naming rule | Run the check in Step 3 |
| `Loaded trials:` lower than expected | Some files failed the naming rule | Run the check in Step 3 |
| `FileNotFoundError: live_motion_model.joblib` | `--use-saved-model` was used before training | Run Step 4 or start without `--use-saved-model` |
| No detected movement yet | Baseline is still forming, trigger thresholds are not crossed, or packets are incomplete | Check the sender is running; use `--probe` to inspect incoming frames |
| Prediction is `nepoznato` | Low SVM confidence, too little movement, or a movement outside the known classes | Inspect confidence, motion thresholds, and whether the window contains the movement |
| Always predicts the same class | Segment boundaries are poor, or `--fps` does not match the CSVs (200 Hz) | Tune trigger thresholds and check `--fps` on both scripts |
| Live and simulated predictions disagree | Model is fine, the captured segment is not | Run `simulate_live_prediction.py` on a real trial to confirm the model |
| `Skipped N incomplete frames` | Frames missing a required object | Expected with `--drop-rate`; with real Vicon it means occlusion |
| `OSError: address already in use` | An old server is still bound | Close the other terminal, or use a different `--port` |

---

## Moving to the real Vicon

Nothing in `vicon_live_capture.py` is specific to the fake sender. Once Nexus
streams to the same port, point it at the capture machine and confirm with:

```powershell
.\.venv\Scripts\python.exe scripts\vicon_live_capture.py --probe --port 51001
```

Two things to verify against the real stream:

1. **Object names.** Probe prints the names Nexus sends. If they are not
   `Left`, `Right`, `Trup`, update `OBJECT_NAME_MAP`.
2. **Frame rate.** Pass the real Nexus rate as `--fps`. Speed and acceleration
   features are scaled by `dt`, so a mismatch silently rescales every feature.

If probe prints nothing while Nexus is streaming, the packet layout differs
from the Vicon UDP Object Stream format assumed by `_parse_binary_packet`.
Capture a raw packet and compare before changing anything else.
