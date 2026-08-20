# regularization_confirm_crossregime.py
# ------------------------------------------------------------
# regularization_search.py(within-2024, 25만행 규모)에서 very_strong_reg가
# baseline보다 holdout score가 나았다(194.5 -> 233.8). 표본이 작아 노이즈일
# 수 있으므로, 더 큰/더 신뢰도 높은 cross-regime holdout(2019-2023 학습 ->
# 2024 전체 검증, 147만행 규모, time_holdout_check.py와 동일 설정)으로
# baseline(460.2) 대비 재확인한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
BASELINE_BRIER = 0.249446

CANDIDATES = {
    "current_tuned (baseline, reference score=460.2)": dict(
        num_leaves=320, min_data_in_leaf=800, feature_fraction=0.5,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_depth=-1),
    "very_strong_reg": dict(
        num_leaves=31, min_data_in_leaf=5000, feature_fraction=0.25,
        bagging_fraction=0.6, bagging_freq=1, lambda_l2=80.0, lambda_l1=10.0, max_depth=5),
}


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()
    print(f"fit={len(fit_df)}  holdout={len(holdout_df)}")

    snapshot_tables = te.build_train_snapshot_tables(fit_df)
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]

    results = []
    for name, params in CANDIDATES.items():
        print(f"\n=== {name} ===")
        models, oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features,
                                    params_override=params, verbose_fold=False)
        oof_cal_brier = brier_score_loss(y_fit, IsotonicRegression(out_of_bounds="clip").fit(oof, y_fit).predict(oof))

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)

        holdout_pred = np.mean(
            [m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
        holdout_brier = brier_score_loss(y_holdout, holdout_pred)
        holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred))

        print(f"  fit OOF(calibrated) score={score(oof_cal_brier):.1f}")
        print(f"  holdout(raw) score={score(holdout_brier):.1f}   holdout(calibrated) score={score(holdout_cal_brier):.1f}")
        results.append((name, score(oof_cal_brier), score(holdout_brier), score(holdout_cal_brier)))

    print("\n\n=== SUMMARY (cross-regime, 2019-2023 -> 2024) ===")
    print(f"{'candidate':50s} {'fit_score':>12s} {'holdout(raw)':>14s} {'holdout(cal)':>14s}")
    for name, a, b, c in results:
        print(f"{name:50s} {a:12.1f} {b:14.1f} {c:14.1f}")


if __name__ == "__main__":
    main()
