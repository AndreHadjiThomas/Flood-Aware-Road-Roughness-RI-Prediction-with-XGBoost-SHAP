#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
RoughCast: Flood‑Aware Road Roughness (RI) Prediction with XGBoost + SHAP

- Clean preprocessing (categoricals → codes; missing M&R → 'No Renovation')
- Two evaluation modes: intra-year (stratified split) and cross-year (2019→2023)
- Optional hyperparameter optimization (BayesianOptimization or Optuna)
- Metrics (RMSE, MSE, R^2), SHAP explainability, and artifact saving
\"\"\"
import os
import json
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import shap

# Optional HPO libs
try:
    from bayes_opt import BayesianOptimization
except Exception:
    BayesianOptimization = None

try:
    import optuna
except Exception:
    optuna = None

# -----------------------------
# Utils
# -----------------------------

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # M&R
    if "M&R_After_6_2019_to_4_2023" in df.columns:
        s = df["M&R_After_6_2019_to_4_2023"].astype("category")
        if "No Renovation" not in s.cat.categories:
            s = s.cat.add_categories("No Renovation")
        df["M&R_After_6_2019_to_4_2023"] = s.fillna("No Renovation")
        df["M&R_code"] = df["M&R_After_6_2019_to_4_2023"].astype("category").cat.codes
        df.drop(columns=["M&R_After_6_2019_to_4_2023"], inplace=True, errors="ignore")
    # Pavement type
    if "Pavement type" in df.columns:
        df["Pavement type_code"] = df["Pavement type"].astype("category").cat.codes
        df.drop(columns=["Pavement type"], inplace=True, errors="ignore")
    # Functional Class
    if "Functional Class" in df.columns:
        df["Functional Class_code"] = df["Functional Class"].astype("category").cat.codes
        df.drop(columns=["Functional Class"], inplace=True, errors="ignore")
    return df

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def make_features(
    df: pd.DataFrame,
    target_col: str,
    drop_id_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    df = prep_categoricals(df)
    df = df.dropna().reset_index(drop=True)
    if drop_id_cols:
        for c in drop_id_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    if feature_cols is None:
        # All except target
        X = df.drop(columns=[target_col])
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    # Bin target into 5 quantiles for stratification
    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y_binned))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def base_xgb(params_overrides=None) -> XGBRegressor:
    base = dict(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.8,
        gamma=1.5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    if params_overrides:
        base.update(params_overrides)
    return XGBRegressor(**base)

# -----------------------------
# HPO
# -----------------------------

def hpo_bayesopt(X_train, y_train, init_points=5, n_iter=25, random_state=42):
    if BayesianOptimization is None:
        raise ImportError("bayesian-optimization is not installed. Add it or drop --use_bayesopt.")
    def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, gamma):
        params = dict(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            gamma=float(gamma),
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist"
        )
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
        return scores.mean()
    pbounds = {
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "gamma": (0.0, 5.0),
    }
    opt = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=random_state, verbose=2)
    opt.maximize(init_points=init_points, n_iter=n_iter)
    best = opt.max["params"]
    best["max_depth"] = int(best["max_depth"])
    return best

def hpo_optuna(X_train, y_train, n_trials=30, random_state=42):
    if optuna is None:
        raise ImportError("optuna is not installed. Add it or drop --use_optuna.")
    def objective(trial):
        params = dict(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist"
        )
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
        return -scores.mean()
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    return best

# -----------------------------
# SHAP plots
# -----------------------------

def shap_plots(model, X_train, X_test, out_dir, waterfall_idx=None):
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)
    explainer = shap.Explainer(model, X_train)
    sv = explainer(X_test, check_additivity=False)

    plt.figure()
    shap.summary_plot(sv, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if waterfall_idx is not None and 0 <= waterfall_idx < len(X_test):
        try:
            shap.plots.waterfall(sv[waterfall_idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"shap_waterfall_idx{waterfall_idx}.png"), dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Waterfall plot failed: {e}")

# -----------------------------
# Main pipeline
# -----------------------------

def run(args):
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "plots"))

    # Load data
    df_2019 = load_df(args.file_2019)
    if args.eval_mode == "cross":
        if not args.file_2023:
            raise ValueError("--file_2023 is required for eval_mode=cross")
        df_2023 = load_df(args.file_2023)
    else:
        df_2023 = None

    # Optional fixed feature list for cross-year (matches notebook)
    fixed_feats = None
    if args.feature_cols:
        fixed_feats = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    drop_ids = [c.strip() for c in args.drop_id_cols.split(",")] if args.drop_id_cols else []

    if args.eval_mode == "cross":
        # Train on 2019, test on 2023 with optional fixed feature set
        X_train, y_train = make_features(df_2019, args.target_col, drop_id_cols=drop_ids, feature_cols=fixed_feats)
        X_test, y_test   = make_features(df_2023, args.target_col, drop_id_cols=drop_ids, feature_cols=fixed_feats)
    else:
        # Intra-year on 2019
        X_all, y_all = make_features(df_2019, args.target_col, drop_id_cols=drop_ids, feature_cols=fixed_feats)
        X_train, X_test, y_train, y_test = stratified_split(X_all, y_all, test_size=args.test_size, random_state=args.random_state)

    # HPO
    best_overrides = {}
    if args.use_bayesopt and args.use_optuna:
        raise ValueError("Choose only one: --use_bayesopt OR --use_optuna")
    if args.use_bayesopt:
        best_overrides = hpo_bayesopt(X_train, y_train, init_points=args.bayes_init, n_iter=args.bayes_iter, random_state=args.random_state)
    elif args.use_optuna:
        best_overrides = hpo_optuna(X_train, y_train, n_trials=args.optuna_trials, random_state=args.random_state)

    # Merge with base params and user overrides
    for k in ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "gamma"]:
        v = getattr(args, k)
        if v is not None:
            best_overrides[k] = v

    model = base_xgb(best_overrides)
    model.fit(X_train, y_train)

    # Predict & metrics
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": rmse(y_test, y_pred),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "eval_mode": args.eval_mode
    }
    print(f"RMSE: {metrics['rmse']:.4f} | MSE: {metrics['mse']:.4f} | R^2: {metrics['r2']:.4f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=True)
    print("Top features:\n", imp.head(20))

    # Save predictions with dataset label
    df_preds_train = X_train.copy()
    df_preds_train["target"] = y_train.values
    df_preds_train["pred"] = model.predict(X_train)
    df_preds_train["dataset"] = "train"

    df_preds_test = X_test.copy()
    df_preds_test["target"] = y_test.values
    df_preds_test["pred"] = y_pred
    df_preds_test["dataset"] = "test"

    preds = pd.concat([df_preds_train, df_preds_test]).reset_index(drop=True)
    preds.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP
    if not args.skip_shap:
        shap_plots(model, X_train, X_test, args.out_dir, waterfall_idx=args.shap_waterfall_idx)

def build_argparser():
    p = argparse.ArgumentParser(description="Flood‑aware Road RI prediction with XGBoost + SHAP")
    p.add_argument("--file_2019", type=str, required=True, help="Path to 2019 CSV (training or full intra-year dataset).")
    p.add_argument("--file_2023", type=str, default=None, help="Path to 2023 CSV (required for eval_mode=cross).")
    p.add_argument("--eval_mode", choices=["cross", "intra"], default="cross", help="cross: 2019→2023; intra: stratified split on 2019.")
    p.add_argument("--target_col", type=str, default="Roughness\nIndex (RI)", help="Target column name.")
    p.add_argument("--drop_id_cols", type=str, default="GISID", help="Comma-separated ID cols to drop if present.")
    p.add_argument("--feature_cols", type=str, default=None, help="Comma-separated list to force a feature set. If omitted, uses all remaining features.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test size for intra mode.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory for artifacts.")

    # Base model overrides
    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample_bytree", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)

    # HPO switches
    p.add_argument("--use_bayesopt", action="store_true", help="Use BayesianOptimization for HPO.")
    p.add_argument("--bayes_init", type=int, default=5, help="BayesOpt random init points.")
    p.add_argument("--bayes_iter", type=int, default=25, help="BayesOpt iterations.")
    p.add_argument("--use_optuna", action="store_true", help="Use Optuna for HPO.")
    p.add_argument("--optuna_trials", type=int, default=30, help="Optuna trials.")

    # SHAP
    p.add_argument("--skip_shap", action="store_true", help="Skip SHAP computation/plots.")
    p.add_argument("--shap_waterfall_idx", type=int, default=None, help="Index of test sample to plot waterfall for.")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
