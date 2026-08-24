# Live Generalization Plot Documentation

These plots document the experiment where the live fPCA/SVM model is trained
only on Anja + Petar recordings, with one held-out Anja/Petar recording per
class, and then tested externally on Lazar recordings that are never used for
training.

Generated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_live_generalization_plots.py
```

Two Petar recordings are skipped because they contain only one frame and cannot
be used for velocity/acceleration features:

```text
data2026\Petar\Snimanje\Podizanje_desna10.csv
data2026\Petar\Snimanje\Sirenje02.csv
```

## 01_dataset_split_counts.png

Image:

```text
results\plots\live_generalization\01_dataset_split_counts.png
```

What it shows:

- How many trials are used for training, Anja/Petar held-out testing, and Lazar
  external testing.
- Counts are stacked by class: `guranje`, `podizanje_desna`, `sirenje`.

What it is for:

- Confirms that Lazar recordings are not included in training.
- Confirms that each test split contains all three classes.
- Makes the class balance visible before interpreting accuracy.

Conclusion:

- Training uses 64 Anja/Petar recordings.
- Held-out Anja/Petar test uses 3 recordings, one per class.
- Lazar external test uses 6 recordings, two per class.

## 02_confusion_anja_petar_heldout.png

Image:

```text
results\plots\live_generalization\02_confusion_anja_petar_heldout.png
```

What it shows:

- Confusion matrix for the three Anja/Petar recordings that were not used for
  training.
- Rows are true labels and columns are predicted labels.

What it is for:

- Checks whether the model can recognize unseen Anja/Petar examples after
  training on the rest of Anja/Petar.

Conclusion:

- Accuracy is 1.000 on this small held-out split.
- Each class has one correct prediction and no confusion with another class.

## 03_confusion_lazar_external.png

Image:

```text
results\plots\live_generalization\03_confusion_lazar_external.png
```

What it shows:

- Confusion matrix for Lazar recordings.
- Lazar recordings are external test data and are not used in training.

What it is for:

- Tests whether the model trained on Anja + Petar transfers to a new person.
- This is the most useful plot for the question "can the model recognize
  Lazar's movements without seeing Lazar in training?"

Conclusion:

- Accuracy is 1.000 on Lazar's 6 valid recordings.
- All two `guranje`, two `podizanje_desna`, and two `sirenje` examples are
  classified correctly.

## 04_fpca_score_space_generalization.png

Image:

```text
results\plots\live_generalization\04_fpca_score_space_generalization.png
```

What it shows:

- First two fPCA component scores for the training trials, held-out Anja/Petar
  trials, and Lazar external trials.
- Color represents the true class.
- Marker shape represents the split:
  - circle: training
  - square: Anja/Petar held-out test
  - triangle: Lazar external test

What it is for:

- Visually checks whether unseen test trials land near the training cluster of
  their true movement class.
- Helps explain why the classifier succeeds or fails.

Conclusion:

- The three movement classes form visibly separated regions in the first two
  fPCA components.
- Lazar's test points fall near the corresponding training regions, which
  supports the 1.000 external-test accuracy.

## 05_prediction_table.png

Image:

```text
results\plots\live_generalization\05_prediction_table.png
```

What it shows:

- One row per non-training trial.
- Columns show split, trial name, true label, predicted label, and result.

What it is for:

- Gives a compact audit trail for the exact examples used in held-out and
  external testing.
- Makes it easy to see which specific file failed if a future run produces a
  misclassification.

Conclusion:

- All held-out Anja/Petar and Lazar examples are classified correctly in this
  run.

## 06_fpca_component_accuracy.png

Image:

```text
results\plots\live_generalization\06_fpca_component_accuracy.png
```

Data table:

```text
results\plots\live_generalization\fpca_component_comparison.csv
```

What it shows:

- Accuracy for fPCA/SVM models trained with 2, 3, 4 and 5 fPCA components.
- The split is the same as the external-test experiment:
  - train: Anja + Petar
  - held-out test: one Anja/Petar recording per class
  - external test: all Lazar recordings

What it is for:

- Checks whether increasing the number of fPCA components improves recognition
  on unseen data.

Conclusion:

- 2, 3, 4 and 5 components all achieve 1.000 accuracy on both test sets in this
  run.
- The live model is now set to 3 fPCA components because the 3rd component
  substantially increases retained variance while keeping the same test
  accuracy.

## 07_fpca_component_variance.png

Image:

```text
results\plots\live_generalization\07_fpca_component_variance.png
```

What it shows:

- Cumulative explained variance ratio for 2, 3, 4 and 5 fPCA components.

What it is for:

- Shows how much additional signal variation is retained when more fPCA
  components are allowed.
- Helps interpret the accuracy result: extra components may explain more
  variation even when they do not improve classification.

Conclusion:

- 2 components explain about 0.538 of the variance.
- 3 components explain about 0.705.
- 4 components explain about 0.759.
- 5 components explain about 0.800.
- The biggest gain after the 2-component setup is the 3rd component, so 3
  components are a useful compromise between simplicity and retained signal
  variation.

## 08_fpca_accuracy_and_variance.png

Image:

```text
results\plots\live_generalization\08_fpca_accuracy_and_variance.png
```

What it shows:

- Left side: held-out Anja/Petar and Lazar external-test accuracy for 2, 3, 4
  and 5 fPCA components.
- Right side: cumulative explained variance for the same component counts.
- The dashed red line marks the selected live-model setting: 3 components.

What it is for:

- Puts the two decision criteria in one place: classification accuracy and
  retained signal variation.
- Makes it clear that the selected value is not based on accuracy alone.

Conclusion:

- Accuracy is unchanged at 1.000 for all tested component counts.
- Moving from 2 to 3 components increases cumulative explained variance from
  about 0.538 to about 0.705.
- 4 and 5 components retain more variance, but with smaller additional gains
  and no accuracy improvement in this test.
- Based on this comparison, 3 components are the current default.
