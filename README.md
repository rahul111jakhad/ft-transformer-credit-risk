# Deep Learning for Credit Risk: Evaluating Feature Tokenizer Transformers Against Industry Baselines

> Code repository for an in-progress paper. **Draft stage** — not yet
> submitted, results subject to change.
>
> **Authors:** Rahul Jakhad, Nikhil Mittal, Vaibhav Saxena, Charchit Bahl

## Abstract

The evolution of credit risk modeling has transitioned from traditional
statistical methods to high-performance tree-based ensembles. However, these
models rely on axis-aligned, discrete partitions that may struggle to
represent complex, smooth risk surfaces. This paper investigates the
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

## Notebooks

The repo is organized as **two notebooks per dataset** — one for the
classical / boosted baselines, one for FT-Transformer — sharing identical
preprocessing so results are directly comparable. All notebooks live under
`notebooks/`.

| # | Notebook | Dataset | Models | Status |
| :- | :--- | :--- | :--- | :--- |
| 01 | [`notebooks/01_lending_club_baselines.ipynb`](notebooks/01_lending_club_baselines.ipynb) | Lending Club | LR / RF / XGBoost | ✅ |
| 02 | [`notebooks/02_home_credit_baselines.ipynb`](notebooks/02_home_credit_baselines.ipynb) | Home Credit | LR / RF / XGBoost | ✅ |
| 03 | `notebooks/03_lending_club_ft_transformer.ipynb` | Lending Club | FT-Transformer | 🚧 |
| 04 | `notebooks/04_home_credit_ft_transformer.ipynb` | Home Credit | FT-Transformer | 🚧 |

[![Open 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul111jakhad/ft-transformer-credit-risk/blob/main/notebooks/01_lending_club_baselines.ipynb)
[![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahul111jakhad/ft-transformer-credit-risk/blob/main/notebooks/02_home_credit_baselines.ipynb)

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
| FT-Transformer | rtdl-revisiting-models | Yes | Optuna (forthcoming) |

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

## Repository layout

```
.
├── notebooks/
│   ├── 01_lending_club_baselines.ipynb     # LR / RF / XGBoost on Lending Club
│   ├── 02_home_credit_baselines.ipynb      # LR / RF / XGBoost on Home Credit
│   ├── 03_lending_club_ft_transformer.ipynb   # forthcoming
│   └── 04_home_credit_ft_transformer.ipynb    # forthcoming
├── data/                               # Raw CSVs (not committed)
│   ├── accepted_2007_to_2018Q4.csv
│   └── home_credit_default.csv
├── artifacts/                          # Generated by notebooks (not committed)
│   ├── lending_club/
│   └── home_credit/
├── results/                            # Publishable outputs (committed)
│   ├── summary_test_metrics.csv
│   └── shap_*.png
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

Each baseline notebook follows the same 10-section structure so they read
side-by-side:

1. Setup and imports
2. Data loading
3. Domain-specific cleaning + target definition
4. Leakage-free preprocessing pipeline (split → filter → encode → scale)
5. Evaluation metric helper
6. Logistic Regression (default → tuned)
7. Random Forest (default → tuned)
8. XGBoost (default → tuned, GPU)
9. SHAP feature importance for XGBoost and RF
10. Test-set summary table

The FT-Transformer notebooks will reuse sections 1–5 verbatim and replace
sections 6–10 with the transformer pipeline so that splits and preprocessing
match the baselines exactly.

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

The four of us are working on this paper. To keep the repo history clean:

* **One branch per workstream**, e.g. `baselines/lending-club-fixes`,
  `ft-transformer/home-credit`, `paper/abstract`. Don't push directly to
  `main`.
* **Pull requests with one review** before merging. Branch protection on
  `main` enforces this.
* **Run notebooks top-to-bottom before committing** so cell-execution
  numbers are sequential. Cleared / re-executed outputs cause noisy diffs.
* **Don't commit data.** `.gitignore` excludes `data/` and `artifacts/`.
  Anything you want to publish goes under `results/`.
* **Pin random seeds.** All notebooks set `RANDOM_SEED = 42` and pass it to
  Optuna, sklearn, and torch. Don't override locally without flagging it.
* **Tag releases for paper milestones**, e.g. `v0.1-draft`,
  `v1.0-submission`, `v1.1-revision`.

## License

MIT (see [`LICENSE`](LICENSE)).
