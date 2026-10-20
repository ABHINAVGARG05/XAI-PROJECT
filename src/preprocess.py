"""Feature engineering and temporal train/val/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import PROCESSED_DIR, RAW_CSV, TARGET

SECONDS_PER_DAY = 86_400
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SPLIT_NAMES = ("train", "val", "test", "test_early", "test_mid", "test_late")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform Amount, cyclic-encode hour of day, drop raw Time/Amount."""
    out = df.sort_values("Time", kind="stable")
    hour = (out["Time"] % SECONDS_PER_DAY) / 3600
    out["Amount_log"] = np.log1p(out["Amount"])
    out["Time_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Time_cos"] = np.cos(2 * np.pi * hour / 24)
    return out.drop(columns=["Time", "Amount"])


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split in time order (no shuffle): 70% train, 15% val, 15% test in 3 cohorts."""
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    test = df.iloc[val_end:]

    third = len(test) // 3
    return {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": test,
        "test_early": test.iloc[:third],
        "test_mid": test.iloc[third : 2 * third],
        "test_late": test.iloc[2 * third :],
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""
    return df.drop(columns=[TARGET]), df[TARGET]


def load_and_split(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Load the raw CSV, engineer features, split temporally and persist as parquet.

    The original row index is preserved in each split so transactions can be
    traced back to the raw file. With ``use_cache`` the parquet files in
    ``data/processed/`` are reused when all of them exist.
    """
    paths = {name: PROCESSED_DIR / f"{name}.parquet" for name in SPLIT_NAMES}
    if use_cache and all(p.exists() for p in paths.values()):
        return {name: pd.read_parquet(p) for name, p in paths.items()}

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Download it with: "
            "kaggle datasets download mlg-ulb/creditcardfraud -p data/raw --unzip"
        )

    splits = temporal_split(engineer_features(pd.read_csv(RAW_CSV)))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(paths[name])
    return splits


if __name__ == "__main__":
    for name, frame in load_and_split(use_cache=False).items():
        print(f"{name:<11} rows={len(frame):>7,}  fraud={int(frame[TARGET].sum()):>4}")
