"""Weather-yield alignment, feature engineering, and baseline models."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    yield_path: Path
    weather_path: Path
    soil_path: Path | None
    out_dir: Path
    target_col: str = "yield"
    crop_col: str = "crop"
    year_col: str = "year"
    state_col: str = "state"
    season_col: str = "season"
    tbase: float = 10.0
    hot_threshold: float = 35.0
    hot_avg_threshold: float = 30.0
    rain_threshold: float = 1.0
    rain_stress_threshold: float = 500.0
    period_days: int = 120
    test_size: float = 0.2


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_and_align(config: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    yield_df = strip_string_columns(clean_columns(pd.read_csv(config.yield_path)))
    weather_df = strip_string_columns(clean_columns(pd.read_csv(config.weather_path)))
    yield_df = ensure_numeric(yield_df, [config.year_col, config.target_col, "area", "production", "fertilizer", "pesticide"])
    weather_df = ensure_numeric(weather_df, [c for c in weather_df.columns if c != config.state_col])
    if config.state_col in weather_df.columns:
        weather_df[config.state_col] = weather_df[config.state_col].astype(str).str.strip()

    merge_keys = [config.state_col, config.year_col]
    missing = [k for k in merge_keys if k not in yield_df.columns or k not in weather_df.columns]
    if missing:
        raise ValueError(f"Cannot align yield and weather data. Missing keys: {missing}")
    df = yield_df.merge(weather_df, on=merge_keys, how="left", suffixes=("", "_weather"))
    notes.append(f"Aligned yield and weather by {merge_keys}.")

    if config.soil_path and config.soil_path.exists():
        soil_df = strip_string_columns(clean_columns(pd.read_csv(config.soil_path)))
        soil_df = ensure_numeric(soil_df, [c for c in soil_df.columns if c != config.state_col])
        if config.state_col in soil_df.columns:
            soil_df[config.state_col] = soil_df[config.state_col].astype(str).str.strip()
            df = df.merge(soil_df, on=config.state_col, how="left", suffixes=("", "_soil"))
            notes.append("Merged soil data by state.")
    return df, notes


def add_annual_weather_features(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    notes: list[str] = []
    rain_col = find_first_existing(df, ["rain", "rainfall", "rainfall_mm", "total_rainfall_mm", "precipitation", "precip_mm"])
    tavg_col = find_first_existing(df, ["tavg", "tavg_c", "avg_temp", "avg_temp_c", "temperature", "mean_temp"])
    tmax_col = find_first_existing(df, ["tmax", "tmax_c", "max_temp", "max_temp_c", "maximum_temperature"])
    humidity_col = find_first_existing(df, ["humidity", "avg_humidity", "avg_humidity_percent", "relative_humidity"])

    df["RainSum"] = df[rain_col] if rain_col else np.nan
    df["TavgMean"] = df[tavg_col] if tavg_col else np.nan
    notes.append(f"RainSum source: {rain_col}; TavgMean source: {tavg_col}.")

    if tmax_col:
        df["HotDays"] = np.maximum(df[tmax_col] - config.hot_threshold, 0.0)
    elif tavg_col:
        df["HotDays"] = np.maximum(df[tavg_col] - config.hot_avg_threshold, 0.0) * (config.period_days / 5.0)
    else:
        df["HotDays"] = 0.0

    if rain_col:
        df["DrySpell"] = np.maximum(config.rain_stress_threshold - df["RainSum"], 0.0) / max(config.rain_stress_threshold, 1.0) * config.period_days
    else:
        df["DrySpell"] = 0.0

    df["GDD"] = np.maximum(df["TavgMean"] - config.tbase, 0.0) * config.period_days if tavg_col else 0.0
    df["HotStress"] = df["HotDays"] / max(config.period_days, 1)
    df["DroughtStress"] = df["DrySpell"] / max(config.period_days, 1)
    df["Stress"] = 0.5 * df["HotStress"] + 0.5 * df["DroughtStress"]
    if humidity_col:
        df["HumidityMean"] = df[humidity_col]
    notes.append("Built RainSum, TavgMean, HotDays, DrySpell, GDD, HotStress, DroughtStress, and Stress.")
    return df, notes


def add_group_lags(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    df = df.copy()
    sort_cols = [c for c in [config.crop_col, config.season_col, config.state_col, config.year_col] if c in df.columns]
    group_cols = [c for c in [config.crop_col, config.season_col, config.state_col] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if group_cols and config.target_col in df.columns:
        for lag in [1, 2, 3]:
            df[f"lag_yield_{lag}"] = df.groupby(group_cols)[config.target_col].shift(lag)
    return df


def build_feature_columns(df: pd.DataFrame, config: ExperimentConfig) -> tuple[list[str], list[str], list[str]]:
    required_weather = ["RainSum", "TavgMean", "HotDays", "DrySpell", "GDD", "Stress"]
    optional_weather = ["HumidityMean", "HotStress", "DroughtStress"]
    management = [c for c in ["area", "fertilizer", "pesticide"] if c in df.columns]
    soil = [c for c in ["N", "P", "K", "pH"] if c in df.columns]
    lags = [c for c in ["lag_yield_1", "lag_yield_2", "lag_yield_3"] if c in df.columns]
    numeric_cols = [c for c in required_weather + optional_weather + management + soil + lags if c in df.columns]
    categorical_cols = [c for c in [config.crop_col, config.season_col] if c in df.columns]
    weather_focus_cols = [c for c in required_weather + optional_weather if c in df.columns]
    return numeric_cols, categorical_cols, weather_focus_cols


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(y_true) - y_pred)
    denom = max(float(np.sum(np.abs(y_true))), 1e-8)
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "WMAPE": float(np.sum(err) / denom * 100),
        "R2": r2_score(y_true, y_pred),
    }


def run_feature_set(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], target_col: str, out_dir: Path) -> pd.DataFrame:
    clean = df.dropna(subset=numeric_cols + categorical_cols + [target_col]).copy()
    X = clean[numeric_cols + categorical_cols]
    y = clean[target_col].astype(float)
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols), ("cat", make_one_hot_encoder(), categorical_cols)],
        remainder="drop",
    )
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rows = []
    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"model": name, **evaluate_predictions(y_test, pred)})
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(out_dir / "model_metrics.csv", index=False)

    best = fitted["RandomForest"]
    perm = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=-1)
    pd.DataFrame({"feature": X_test.columns, "perm_importance": perm.importances_mean}).sort_values("perm_importance", ascending=False).to_csv(
        out_dir / "permutation_importance.csv", index=False
    )
    return metrics


def run_experiment(config: ExperimentConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    df, notes = load_and_align(config)
    df, feature_notes = add_annual_weather_features(df, config)
    df = add_group_lags(df, config)
    numeric_cols, categorical_cols, weather_focus_cols = build_feature_columns(df, config)
    metrics = run_feature_set(df, numeric_cols, categorical_cols, config.target_col, config.out_dir)

    corr_rows = []
    clean = df.dropna(subset=weather_focus_cols + [config.target_col]).copy()
    for col in weather_focus_cols:
        if clean[col].nunique(dropna=True) > 1:
            pearson = pearsonr(clean[col], clean[config.target_col]).statistic
            spearman = spearmanr(clean[col], clean[config.target_col]).statistic
        else:
            pearson = np.nan
            spearman = np.nan
        corr_rows.append({"feature": col, "pearson": pearson, "spearman": spearman})
    pd.DataFrame(corr_rows).to_csv(config.out_dir / "weather_correlations.csv", index=False)
    (config.out_dir / "experiment_notes.json").write_text(json.dumps({"notes": notes + feature_notes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Outputs saved to: {config.out_dir}")


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(r"D:\sicau\code\CAMA-Yield\india_dataset-1")
    parser = argparse.ArgumentParser(description="Run weather-yield prediction experiment.")
    parser.add_argument("--yield-path", type=Path, default=default_data_dir / "crop_yield.csv")
    parser.add_argument("--weather-path", type=Path, default=default_data_dir / "state_weather_data_1997_2020.csv")
    parser.add_argument("--soil-path", type=Path, default=default_data_dir / "state_soil_data.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_weather_yield"))
    parser.add_argument("--target-col", default="yield")
    parser.add_argument("--period-days", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(args.yield_path, args.weather_path, args.soil_path, args.out_dir, target_col=args.target_col, period_days=args.period_days)
    run_experiment(config)


if __name__ == "__main__":
    main()
