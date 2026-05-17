# Deep Learning for Credit Risk: Evaluating Feature Tokenizer Transformers Against Industry Baselines

> Code and write-up for an empirical study. **Work in progress** —
> results subject to change.
>
> **Authors:** Rahul Jakhad, Nikhil Mittal, Vaibhav Saxena, Charchit Bahl

## Summary

The evolution of credit risk modeling has transitioned from traditional
statistical methods to high-performance tree-based ensembles. However, these
models rely on axis-aligned, discrete partitions that may struggle to
represent complex, smooth risk surfaces. This study investigates the
**Feature Tokenizer Transformer (FT-Transformer)** as a deep learning
alternative for tabular credit data. We evaluate the architecture against
three industry baselines — **Logistic Regression**, **Random Forest**, and
**XGBoost** — across the **Lending Club** and **Home Credit Default Risk**
datasets. For a fair comparison, all models are evaluated under a uniform
experimental setup using identical training data and standardized
pre-processing conditions. Furthermore, we employ **Optuna** for automated
Bayesian hyperparameter tuning to ensure each algorithm achieves its optimal
predictive potential. Performance is analyzed through a multi-dimensional
metric framework including **AUC, Gini, KS, and AUCPR**, alongside a
comparative assessment of training time and architectural hyperparameter
sensitivity.

> **Scope.** This is an empirical study; the repository — code, notebooks,
> and curated results — is the primary deliverable. A written report
> synthesizing the findings can be generated from the committed results once
> all experiments are complete.

## Notebooks

The repo is organized as **two notebooks per dataset** — one for the
classical / boosted baselines, one for FT-Transformer — sharing identical
preprocessing so results are directly comparable. All notebooks live under
`notebooks/`.

| # | Notebook | Dataset | Models | Status |
| :- | :--- | :--- | :--- | :--- |
| 01 | [`notebooks/01_lending_club_baselines.ipynb`](notebooks/01_lending_club_baselines.ipynb) | Lending Club | LR / RF / XGBoost | ✅ |
| 02 | [`notebooks/02_home_credit_baselines.ipynb`](notebooks/02_home_credit_baselines.ipynb) | Home Credit | LR / RF / XGBoost | ✅ |
| 03 | [`notebooks/03_lending_club_ft_transformer.ipynb`](notebooks/03_lending_club_ft_transformer.ipynb) | Lending Club | FT-Transformer (full experiments) | ✅ |
| 04 | `notebooks/04_home_credit_ft_transformer.ipynb` | Home Credit | FT-Transformer | 🚧 |

[![Open 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul111jakhad/ft-transformer-credit-risk/blob/main/notebooks/01_lending_club_baselines.ipynb)
[![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul111jakhad/ft-transformer-credit-risk/blob/main/notebooks/02_home_credit_baselines.ipynb)
[![Open 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul111jakhad/ft-transformer-credit-risk/blob/main/notebooks/03_lending_club_ft_transformer.ipynb)

## Datasets

* **Lending Club** — [Kaggle: All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
  We retain only the **63 application-time features** that are available at
  loan origination; everything tied to repayment behavior, hardship,
  settlement, or post-origination status is removed up front to avoid target
  leakage.
* **Home Credit Default Risk** — [Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk).
  The notebook expects a single CSV containing the joined application table
  plus aggregated supplementary features (mean / count / min / max over
  `bureau`, `previous_application`, `POS_CASH_balance`,
  `credit_card_balance`, `installments_payments`, and `bureau_balance`)
  produced by an upstream ETL notebook.

## Models compared

| Model | Library | GPU | Tuning |
| :--- | :--- | :--- | :--- |
| Logistic Regression | scikit-learn | No | Optuna (20 trials) |
| Random Forest | scikit-learn | No | Optuna (10 trials) |
| XGBoost | xgboost | Yes | Optuna (30 trials) |
| FT-Transformer | rtdl-revisiting-models | Yes | Optuna (30 trials) |

For each model we report performance with **default hyperparameters** and
with **Optuna-tuned hyperparameters**, on train / validation / test splits.

## Methodology

* **Stratified 64 / 16 / 20 splits** on the binary target.
* **Leakage control.** Every transformer (median imputer, label encoder,
  standard scaler) is fit on the training split only; validation and test
  splits are transformed with the train-fit objects. Highly correlated
  numeric features and high-missingness columns are dropped using statistics
  computed only on the training set.
* **Hyperparameter tuning** uses Optuna (TPE sampler, seeded for
  reproducibility) with validation AUC as the objective. Search spaces are
  kept conservative (depth 2–4 for trees, log-uniform learning rates) to
  discourage overfitting on these imbalanced datasets.
* **`scale_pos_weight`** for XGBoost is searched in a band centered on the
  empirical imbalance ratio, $\text{neg}/\text{pos}$.
* **Reporting metrics**: AUC, Gini, KS, AUCPR, plus top-decile Precision /
  Recall (the riskiest 10% of scored applications) — closer to how a credit
  decision system actually consumes scores.
* **Reproducibility tracking.** Every notebook writes through `src/tracking.py`:
  per-model performance CSVs, tuned hyperparameters as JSON, full Optuna trial
  history, test-set predictions for cross-model comparison, wall-clock timings
  (default train / tuned train / tuning loop / inference latency), and a
  one-time environment snapshot (Python/library versions, GPU info, seed).
  Test-set summaries are accumulated in `results/summary_test_metrics_<dataset>.csv`
  with append-and-dedupe semantics — re-running a model overwrites its row,
  adding a new model appends.
* **FT-Transformer analyses.** Beyond the headline metrics, the FT-Transformer
  notebook includes: an imbalance-handling study (plain vs weighted BCE vs
  focal loss), a categorical-tokenizer ablation, hyperparameter sensitivity
  sweeps, a sample-size study, an MLP baseline, attention-map extraction
  (per-feature and per-head, compared against XGBoost SHAP rankings), and a
  decision-boundary analysis that contrasts the axis-aligned, piecewise-constant
  surface of tree ensembles against the smooth surface of FT-Transformer.

## Repository layout

```
.
├── src/                                # Shared preprocessing + evaluation + tracking
│   ├── __init__.py
│   ├── preprocessing.py                # Pipeline: split, filter, encode, scale
│   ├── datasets.py                     # Dataset-specific loaders / cleaners
│   ├── evaluation.py                   # AUC / Gini / KS / AUCPR / top-decile
│   ├── interpretation.py               # SHAP helpers for tree models
│   ├── tracking.py                     # Logger, timing, artifact persistence
│   └── ft_transformer_utils.py         # PyTorch training loop + FTTransformerWrapper
├── notebooks/
│   ├── 01_lending_club_baselines.ipynb     # LR / RF / XGBoost on Lending Club
│   ├── 02_home_credit_baselines.ipynb      # LR / RF / XGBoost on Home Credit
│   ├── 03_lending_club_ft_transformer.ipynb   # FT-Transformer + full experiments
│   └── 04_home_credit_ft_transformer.ipynb    # forthcoming
├── data/                               # Raw CSVs (NOT committed)
│   ├── accepted_2007_to_2018Q4.csv
│   └── home_credit_default.csv
├── artifacts/                          # Per-run outputs (NOT committed)
│   ├── lending_club/
│   │   ├── {model}_perf.csv                  # train/valid/test metrics
│   │   ├── {model}_best_params.json          # tuned hyperparameters
│   │   ├── {model}_study.csv                 # Optuna trial history
│   │   ├── {model}_predictions_test.npy      # for cross-model comparison
│   │   ├── y_test.npy                        # shared across all models
│   │   ├── timings.csv                       # default/tuned train, tuning, inference
│   │   ├── environment.json                  # versions + GPU info
│   │   ├── shap_importance_*.csv
│   │   └── shap_*.png
│   └── home_credit/                          # same structure
├── results/                            # Curated summary tables (COMMITTED)
│   ├── summary_test_metrics_lending_club.csv
│   ├── summary_test_metrics_home_credit.csv
│   └── (figures for the write-up)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

All preprocessing, evaluation, and tracking logic lives in `src/` so the
four notebooks (two datasets × two model families) operate on **identical
splits and identical features**. The notebooks themselves contain only
model-specific code (LR/RF/XGBoost training and tuning for the baseline
notebooks, FT-Transformer training and analysis for the FT notebooks).

### `artifacts/` vs `results/`

* **`artifacts/`** — raw per-run outputs, gitignored. Re-generated by
  re-running notebooks. Includes models, predictions, Optuna histories,
  per-run metric CSVs, environment snapshots.
* **`results/`** — curated, write-up-ready outputs, committed to git. The
  cross-run summary tables and figures for the eventual report. Grows
  monotonically: each `save_run_artifacts` call appends (or updates) one
  row per `(dataset, model_name)`.

This separation prevents accidentally committing hundreds of MB of
intermediate state while keeping the curated outputs in version control.

### Notebook structure

The two **baseline notebooks** (01, 02) share the same structure: setup,
data loading, domain cleaning, the shared preprocessing pipeline, then
Logistic Regression / Random Forest / XGBoost (each default → Optuna-tuned),
SHAP feature importance, and a test-set summary table.

The **FT-Transformer notebook** (03) is organized in two parts that run
top-to-bottom in a single notebook:

* **Part 1 — Sanity run (§1–8).** Setup, the shared preprocessing pipeline
  (reused verbatim from the baselines so splits and features match exactly),
  PyTorch DataLoaders, and a single FT-Transformer trained with the original
  FT-Transformer paper's recommended default hyperparameters, to confirm the
  architecture trains end-to-end.
* **Part 2 — Full experiments (§9–19).** Imbalance-handling study,
  categorical-tokenizer ablation, 30-trial Optuna tuning, the final tuned
  model, hyperparameter sensitivity sweeps, a sample-size study, an MLP
  baseline, attention-map extraction, an optional seed-variance study, and a
  decision-boundary analysis.

Because preprocessing lives in `src/`, the FT-Transformer notebook operates
on byte-identical splits and features to the baselines — model comparisons
are fair by construction.

## Running the notebooks

The notebooks resolve `data/` and `artifacts/` relative to the repo root
(one directory above `notebooks/`), so it doesn't matter whether you launch
Jupyter from the repo root or from inside `notebooks/`. On Colab, an
optional cell mounts your Drive and routes everything through `DRIVE_ROOT`.

### Local / server

```bash
git clone https://github.com/rahul111jakhad/ft-transformer-credit-risk.git
cd ft-transformer-credit-risk
pip install -r requirements.txt
# Place the CSVs under ./data/
jupyter notebook
# then open notebooks/01_lending_club_baselines.ipynb
```

### Google Colab

1. Open the notebook via the Colab badge above.
2. Uncomment the three lines in the **"Optional: Google Colab Drive mount"**
   cell near the top.
3. Place the dataset CSVs at
   `MyDrive/credit_risk_modeling/data/<filename>.csv` on your Drive.
4. Run all cells.

The path cell auto-routes everything through `DRIVE_ROOT` when defined, so
no other edits are needed.

### GPU

For GPU XGBoost you need a CUDA build of `xgboost` and a compatible NVIDIA
driver (Colab T4 / A100 work out of the box). On CPU-only machines, change
`device="cuda"` to `device="cpu"` in the XGBoost cells.

## Contributing (for the team)

The four of us are working on this study. To keep the repo history clean:

* **One branch per workstream**, e.g. `baselines/lending-club-fixes`,
  `ft-transformer/home-credit`, `docs/report-draft`. Don't push directly to
  `main`.
* **Pull requests with one review** before merging. Branch protection on
  `main` enforces this.
* **Run notebooks top-to-bottom before committing** so cell-execution
  numbers are sequential. Cleared / re-executed outputs cause noisy diffs.
* **Don't commit data.** `.gitignore` excludes `data/` and `artifacts/`.
  Anything you want to publish goes under `results/`.
* **Pin random seeds.** All notebooks set `RANDOM_SEED = 42` and pass it to
  Optuna, sklearn, and torch. Don't override locally without flagging it.
* **Tag releases for project milestones**, e.g. `v0.1-draft`,
  `v1.0-study-complete`, `v1.1-revision`.

## License

MIT (see [`LICENSE`](LICENSE)).
