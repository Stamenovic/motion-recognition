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

The primary prediction is `fPCA + linear SVM`.

The fallback/check prediction is `statistical features + linear SVM`.

This fits the planned SPACE start/stop workflow because the model receives a
completed movement segment, not an incomplete frame-by-frame stream.

## Train model

```powershell
.\.venv\Scripts\python.exe scripts\train_live_models.py
```

This saves:

```text
models/live_motion_model.joblib
```

The file is ignored by git through the existing `models/*.joblib` rule.

## Evaluate with Leave-One-Out

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_live_models.py
```

This retrains the live-ready model on each Leave-One-Out split and reports both
the fPCA/SVM and statistical/SVM predictions.

## Simulate a completed live segment

```powershell
.\.venv\Scripts\python.exe scripts\simulate_live_prediction.py --trial Sirenje_04
```

This loads one existing CSV trial and classifies it as if it had just been
captured from the live buffer.

## Vicon integration placeholder

`scripts/vicon_live_capture.py` contains the planned loop, but the concrete SDK
calls still need to be added:

- `connect_to_vicon()`
- `read_vicon_frame(client)`
- `space_was_pressed()`

The Vicon frame conversion must provide translations for:

```text
Left:left
Right:right
Trup:trup
```

Each recorded segment is then converted to `TrialRecord` through
`LiveSegmentBuffer`.
