# RoughCast: Flood‑Aware Road Roughness (RI) Prediction with XGBoost + SHAP

Predict **road surface roughness (RI)** from roadway condition and climate/flood factors using **XGBoost**, with:
- Clean preprocessing for categorical columns (M&R, Pavement type, Functional Class)
- Two evaluation modes: **intra-year** (stratified train/test split) and **cross-year** (train 2019 → test 2023)
- Optional **Bayesian hyperparameter optimization** (BayesianOptimization or Optuna)
- Model explainability via **SHAP** (summary + bar + optional waterfall)
- Saved predictions and ranked feature importance CSVs + plots

> Motivation: field collection of IRI/RI is expensive. This repo builds a practical surrogate model from asset condition and flood exposure features.

---

## Quickstart

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run (cross-year: train on 2019, test on 2023)
```bash
python road_ri_forecast_xgb.py   --file_2019 "/path/to/Treated_Dataset_2019_modified.csv"   --file_2023 "/path/to/Treated_Dataset_2023_modified.csv"   --eval_mode cross   --out_dir outputs   --use_bayesopt   --bayes_iter 25
```

### Run (intra-year: stratified split on target quantiles)
```bash
python road_ri_forecast_xgb.py   --file_2019 "/path/to/Treated_Dataset_2019_modified.csv"   --eval_mode intra   --test_size 0.2   --random_state 42   --out_dir outputs
```

### Optional: Optuna instead of BayesianOptimization
```bash
python road_ri_forecast_xgb.py   --file_2019 "/path/to/Treated_Dataset_2019_modified.csv"   --file_2023 "/path/to/Treated_Dataset_2023_modified.csv"   --eval_mode cross   --use_optuna   --optuna_trials 40   --out_dir outputs
```

---

## Data expectations

Your CSVs should contain (common examples from the original notebook):
- Target column (default): `Roughness\nIndex (RI)`  ← note the newline
- ID column (optional): `GISID` (dropped if present)
- Categorical columns (encoded automatically if present):
  - `M&R_After_6_2019_to_4_2023` (missing → `"No Renovation"` then encoded)
  - `Pavement type`
  - `Functional Class`
- Feature examples used in cross-year test:
  - `Surface Distress Index (SDI)`, `LADD`, `Rutting (ACP Only)`, `L&T Crk / Linear Crk`, `NLAD`,
    `Pavement Width (ft)`, `Pavement Length (ft)`, `Ground Deformation`, `Functional Code`,
    `Add Area (yd2)`, `DEPTH_mean`, `Flood Susceptibility`, `Water SPEED`, `DEPTH_max`,
    `water elevation`, `Frequency`

You can rename any column via CLI flags `--target_col` and `--feature_cols` (comma-separated).

---

## Outputs

Inside `--out_dir` (default `outputs/`):
- `metrics.json` — RMSE/MSE/R² (and config)
- `feature_importance.csv` — XGBoost gain importances
- `predictions.csv` — predictions with dataset tag (2019/2023) and features
- `plots/shap_summary.png`, `plots/shap_bar.png`, optional `plots/shap_waterfall_idx{K}.png`

---

## CLI (summary)

Run `python road_ri_forecast_xgb.py -h` for all options. Key flags:

- `--file_2019`, `--file_2023` (paths)
- `--eval_mode` `cross|intra`
- `--test_size` (intra), `--random_state`
- `--target_col` (default `'Roughness\nIndex (RI)'`)
- `--drop_id_cols` (comma list; default `"GISID"`)
- `--feature_cols` (comma list to force a specific feature set; otherwise uses all remaining numeric + encoded features)
- `--use_bayesopt` / `--bayes_iter` **or** `--use_optuna` / `--optuna_trials`
- `--n_estimators` (default 500) and other base params
- `--shap_waterfall_idx` (int; plot one sample waterfall)

---

## License
MIT
