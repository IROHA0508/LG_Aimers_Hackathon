# regime_hypothesis_check.py
# ------------------------------------------------------------
# 가설: 2019~2023 -> 2024 홀드아웃에서 본 대붕괴(score 2660 -> 460)가 "매년
# 조금씩 있는 drift"가 아니라 2024시즌에 생긴 "레짐 전환"(KBO 2024시즌 ABS/
# 피치클락 도입 등 판정·전략 변화 추정) 때문이라면, **같은 레짐 내부에서는**
# 시간 기반 예측이 훨씬 정상적으로 잘 될 것이다.
#
# 검증: season==2024 데이터만 떼서, 그 안에서 다시 시간순으로 쪼갠다.
#   fit    = 2024년 3~7월 (176,611행)
#   holdout= 2024년 8~10월 (76,896행) - "같은 레짐 내의 미래"
# 이 점수가 지난 실험의 cross-regime holdout(2019-2023 -> 2024, score 460.2)보다
# 뚜렷하게 높다면 레짐 전환 가설을 지지하는 증거. 여전히 낮다면 레짐 전환이
# 아니라 다른 종류의(혹은 상시적인) drift라는 뜻.
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


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    s24 = train[train["season"] == 2024].sort_values("row_id").reset_index(drop=True)

    fit_df = s24[s24["game_month"] <= 7].copy()
    holdout_df = s24[s24["game_month"] >= 8].copy()
    print(f"fit(2024, month<=7)={len(fit_df)}  holdout(2024, month>=8)={len(holdout_df)}")
    print(f"fit success_rate={fit_df[TARGET_COL].mean():.4f}  "
          f"holdout success_rate={holdout_df[TARGET_COL].mean():.4f}")

    print("\nBuild snapshot tables from fit period only...")
    snapshot_tables = te.build_train_snapshot_tables(fit_df)

    print("Build fit features (as-of cumulative within fit period)...")
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    print("Build holdout features (static snapshot path, same as script.py)...")
    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]

    print(f"\nn_features={len(feat_cols)}  n_fit={len(X_fit)}  n_holdout={len(X_holdout)}")

    print("\nTrain LightGBM on fit period (5-fold)...")
    lgb_models, fit_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features)
    fit_oof_brier = brier_score_loss(y_fit, fit_oof)
    print(f"[fit period internal OOF] brier={fit_oof_brier:.5f} score={score(fit_oof_brier):.1f}")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(fit_oof, y_fit)
    fit_oof_cal_brier = brier_score_loss(y_fit, iso.predict(fit_oof))
    print(f"[fit period internal OOF, calibrated] brier={fit_oof_cal_brier:.5f} score={score(fit_oof_cal_brier):.1f}")

    holdout_pred_raw = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)
    holdout_raw_brier = brier_score_loss(y_holdout, holdout_pred_raw)
    print(f"\n[holdout Aug-Oct 2024, uncalibrated] brier={holdout_raw_brier:.5f} score={score(holdout_raw_brier):.1f}")

    holdout_pred_cal = iso.predict(holdout_pred_raw)
    holdout_cal_brier = brier_score_loss(y_holdout, holdout_pred_cal)
    print(f"[holdout Aug-Oct 2024, calibrated] brier={holdout_cal_brier:.5f} score={score(holdout_cal_brier):.1f}")

    print("\n=== Comparison ===")
    print(f"cross-regime holdout (2019-2023 -> 2024) from earlier run: score=460.2 (reference)")
    print(f"within-regime holdout (2024 Mar-Jul -> 2024 Aug-Oct): score={score(holdout_cal_brier):.1f}")


if __name__ == "__main__":
    main()
