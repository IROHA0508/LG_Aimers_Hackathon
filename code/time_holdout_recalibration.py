# time_holdout_recalibration.py
# ------------------------------------------------------------
# time_holdout_check.py에서 확인된 문제: fit(2019-2023) 성공률 53.16% ->
# holdout(2024) 성공률 48.61%로 큰 폭 하락. fit 기간 OOF로 학습한 Isotonic
# 보정기를 그대로 holdout에 적용해도 거의 개선이 없음(prior shift를 못 잡음).
#
# 이 스크립트는 "prior probability shift correction"(Saerens/Elkan 방식)을
# 적용해본다: source prior(π_s, fit 기간 전체 평균)에서 target prior(π_t, 다음
# 시즌 추정치)로 예측 확률을 Bayes-consistent하게 재조정한다.
#   p' = (p * r) / (p * r + (1-p) * (1-π_t)/(1-π_s)),  r = π_t/π_s
#
# π_t 추정 방법 두 가지를 비교한다:
#   (A) trend: fit 기간 시즌별 성공률에 선형회귀 -> 다음 시즌으로 외삽 (실전에서도
#       쓸 수 있는 "눈가리고" 방식 - holdout 라벨을 전혀 안 봄)
#   (B) oracle: holdout 실제 평균 성공률을 그대로 사용 (라벨을 봐야 하므로 실전에선
#       못 쓰지만, "prior shift 보정만으로 얻을 수 있는 이론적 상한"을 보여주는 참고용)
#
# 추가로 "최근 시즌만으로 보정" 방식(재보정을 fit 마지막 시즌 OOF로만 fit)도
# 비교한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL

BASELINE_BRIER = 0.249446


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def prior_shift(p, pi_s, pi_t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = pi_t / pi_s
    num = p * r
    den = num + (1 - p) * (1 - pi_t) / (1 - pi_s)
    return num / den


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()

    season_rates = fit_df.groupby("season")[TARGET_COL].mean()
    print("Season success rate (fit period):")
    print(season_rates)

    slope, intercept = np.polyfit(season_rates.index.values, season_rates.values, deg=1)
    pi_t_trend = slope * 2024 + intercept
    pi_s = fit_df[TARGET_COL].mean()
    pi_t_oracle = holdout_df[TARGET_COL].mean()
    print(f"\npi_s (fit overall) = {pi_s:.4f}")
    print(f"pi_t trend-extrapolated (season->rate linreg, no holdout labels used) = {pi_t_trend:.4f}")
    print(f"pi_t oracle (actual holdout mean, reference only) = {pi_t_oracle:.4f}")

    print("\nBuild snapshot tables from fit period only...")
    snapshot_tables = te.build_train_snapshot_tables(fit_df)

    print("Build fit features...")
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    print("Build holdout features (static snapshot path)...")
    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]
    fit_season = fit_df["season"].values

    print(f"\nTrain LightGBM on fit period (5-fold)...")
    lgb_models, fit_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features)

    holdout_pred_raw = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)

    results = {}

    # baseline: isotonic on full fit OOF (기존 방식, time_holdout_check.py와 동일)
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(fit_oof, y_fit)
    pred_iso_full = iso_full.predict(holdout_pred_raw)
    results["baseline: isotonic(fit 전체 2019-2023)"] = pred_iso_full

    # recency: isotonic on last season (2023) OOF only
    mask_last = fit_season == 2023
    iso_recent = IsotonicRegression(out_of_bounds="clip")
    iso_recent.fit(fit_oof[mask_last], y_fit[mask_last])
    pred_iso_recent = iso_recent.predict(holdout_pred_raw)
    results["recency: isotonic(2023 OOF만)"] = pred_iso_recent

    # prior-shift on top of baseline isotonic, trend-extrapolated target
    pred_shift_trend = prior_shift(pred_iso_full, pi_s, pi_t_trend)
    results["prior-shift trend (baseline isotonic + trend pi_t)"] = pred_shift_trend

    # recency + prior-shift trend combined
    pred_recent_shift_trend = prior_shift(pred_iso_recent, y_fit[mask_last].mean(), pi_t_trend)
    results["recency + prior-shift trend"] = pred_recent_shift_trend

    # oracle prior-shift (참고용 상한선 - 실전에선 못 씀)
    pred_shift_oracle = prior_shift(pred_iso_full, pi_s, pi_t_oracle)
    results["[oracle only] prior-shift with true holdout mean"] = pred_shift_oracle

    print("\n=== Holdout(2024) results ===")
    print(f"{'method':55s} {'mean_pred':>10s} {'brier':>10s} {'score':>10s}")
    for name, pred in results.items():
        b = brier_score_loss(y_holdout, pred)
        print(f"{name:55s} {pred.mean():10.4f} {b:10.5f} {score(b):10.1f}")

    print(f"\n(참고) actual holdout mean = {y_holdout.mean():.4f}")


if __name__ == "__main__":
    main()
