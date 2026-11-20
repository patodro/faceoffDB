# Faceoff dismissal classifier baseline

This workspace has manually labeled faceoff clips in `raw_clips/` and a master
annotation CSV at `../OneDrive/annotations.csv`. `train_baseline.py` learns the
three current labels (`CLEAN`, `DISMISSAL`, and `RESET`) from dense optical-flow
features. It is a first benchmark, not a production detector: it uses coarse
whole-frame motion and does not localize the linesman.

## Setup and run

From this directory, install the dependencies into the selected Python
environment and run:

```powershell
python -m pip install -r requirements.txt
python train_baseline.py
```

Run the fast loader/feature checks with:

```powershell
python -m unittest discover -s tests -v
```

If the annotation file or media folders are elsewhere, pass explicit paths:

```powershell
python train_baseline.py `
  --annotations "O:\_PAT\OneDrive\annotations.csv" `
  --clips-root "O:\_PAT\faceoffDB\raw_clips" `
  --reference-root "O:\_PAT\faceoffDB\reference_library"
```

The loader checks labels, duplicate clip rows, and clip paths before extracting
features. It prefers a matching trimmed clip in the label-specific
`reference_library/` folder, then checks other reference folders if needed. It
uses `raw_clips/<game_id>/` only when no reference-library clip with that name
exists. The annotation label takes precedence over the reference-library folder
name; this accommodates clips whose curated folder does not match the master
annotation.

## Evaluation and outputs

Evaluation holds out one entire game at a time (leave-one-game-out
cross-validation), then reports per-class precision/recall/F1 and a confusion
matrix. This is more meaningful than randomly splitting clips, which could put
clips from the same broadcast/game in both training and test sets. With only a
few games and 20 dismissal labels, these scores are preliminary and likely
high-variance.

The script writes:

- `artifacts/faceoff_baseline.joblib` — classifier and feature configuration
- `artifacts/grouped_cv_predictions.csv` — held-out predictions joined with
  `game_id`, clip filename, and annotated `linesman_id`
- `artifacts/unreadable_clips.csv` — clips that could not be decoded; these are
  explicitly excluded from training and evaluation

The linesman field is carried through from the annotations; this baseline does
not visually identify officials. A later stage can scan full broadcasts for
candidate faceoffs, then apply the classifier to those clips.
