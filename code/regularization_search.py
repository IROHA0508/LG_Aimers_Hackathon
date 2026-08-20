# regularization_search.py
# ------------------------------------------------------------
# 지금까지 쓴 LGB 하이퍼파라미터(num_leaves=320, min_data_in_leaf=800,
# feature_fraction=0.5, lambda_l2=10.0)는 랜덤 K-fold CV를 최소화하도록
# 튜닝된 값이다. 이번에 이 값이 forward-time(미래 시점) 예측에서도 최선인지
# 재검증한다 - 더 강하게 정규화한 후보들과 비교.
#
# 빠른 반복을 위해 within-2024 harness(regime_hypothesis_check.py와 동일한
# fit=2024 3~7월/holdout=2024 8~10월 분할, 25만행 규모)를 사용한다. 여기서
# 최선인 후보를 추리면, 마지막에 원래 cross-regime(2019-2023->2024, 147만행)
# 홀드아웃으로 재확인한다.
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
    "current_tuned (baseline)": dict(
        num_leaves=320, min_data_in_leaf=800, feature_fraction=0.5,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_depth=-1),
    "mild_reg": dict(
        num_leaves=127, min_data_in_leaf=1500, feature_fraction=0.4,
        bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, max_depth=8),
    "strong_reg": dict(
        num_leaves=63, min_data_in_leaf=3000, feature_fraction=0.3,
        bagging_fraction=0.6, bagging_freq=1, lambda_l2=40.0, lambda_l1=5.0, max_depth=6),
    "very_strong_reg": dict(
        num_leaves=31, min_data_in_leaf=5000, feature_fraction=0.25,
        bagging_fraction=0.6, bagging_freq=1, lambda_l2=80.0, lambda_l1=10.0, max_depth=5),
}


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    s24 = train[train["season"] == 2024].sort_values("row_id").reset_index(drop=True)

    fit_df = s24[s24["game_month"] <= 7].copy()
    holdout_df = s24[s24["game_month"] >= 8].copy()
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
        print(f"\n=== {name}: {params} ===")
        models, oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features,
                                    params_override=params, verbose_fold=False)
        oof_brier = brier_score_loss(y_fit, oof)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)
        oof_cal_brier = brier_score_loss(y_fit, iso.predict(oof))

        holdout_pred = np.mean(
            [m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
        holdout_brier = brier_score_loss(y_holdout, holdout_pred)
        holdout_cal_pred = iso.predict(holdout_pred)
        holdout_cal_brier = brier_score_loss(y_holdout, holdout_cal_pred)

        n_leaves_used = [m.num_trees() for m in models]
        print(f"  fit OOF brier={oof_brier:.5f} score={score(oof_brier):.1f}  "
              f"(calibrated: brier={oof_cal_brier:.5f} score={score(oof_cal_brier):.1f})")
        print(f"  holdout brier={holdout_brier:.5f} score={score(holdout_brier):.1f}  "
              f"(calibrated: brier={holdout_cal_brier:.5f} score={score(holdout_cal_brier):.1f})")
        print(f"  avg n_trees(early-stopped)={np.mean(n_leaves_used):.0f}")

        results.append((name, oof_cal_brier, holdout_brier, holdout_cal_brier))

    print("\n\n=== SUMMARY ===")
    print(f"{'candidate':30s} {'fit_oof_score':>14s} {'holdout_score(raw)':>20s} {'holdout_score(cal)':>20s}")
    for name, oof_cal_brier, holdout_brier, holdout_cal_brier in results:
        print(f"{name:30s} {score(oof_cal_brier):14.1f} {score(holdout_brier):20.1f} {score(holdout_cal_brier):20.1f}")


if __name__ == "__main__":
    main()
