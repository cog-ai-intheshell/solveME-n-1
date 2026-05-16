# SolveMe - Triangle ML Test

Small recruitment test around triangle image classification.

The goal is to build a complete ML pipeline to predict whether an image contains a right triangle (`1`) or a non-right triangle (`0`), then compare several approaches rigorously.

## Files

- `solveme.ipynb`: main test notebook.
- `gen-images.py`: image dataset generation script.
- `Experience-n°1/`: draft / side experiment.

## Dataset Generation

From the project root:

```bash
python gen-images.py
```

This creates the folder:

```text
triangle_images/
  0/
  1/
```

Default parameters:

- `N_SAMPLES = 20000`
- `RECTANGLE_PCT = 0.3`
- `IMAGE_SIZE = 256`
- `SEED = 42`

## Suggested Dependencies

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn xgboost hdbscan optuna shap torch torchvision tqdm
```

## Expected Work

The candidate must complete and execute `solveme.ipynb`.

All sections are mandatory:

- data analysis;
- clustering and PCA;
- leakage-free splits;
- custom metrics;
- simple baselines;
- XGBoost, CNN, GAN, world model;
- calibration, thresholds, and final evaluation;
- SHAP and error analysis;
- final best model;
- manual test with one right triangle and one non-right triangle drawn by the candidate.

## Expected Deliverables

- executed notebook with visible outputs;
- model comparison table;
- final predictions;
- split indices;
- short conclusion report;
- best-model artifacts when applicable.
