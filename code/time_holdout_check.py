# time_holdout_check.py
# ------------------------------------------------------------
# 목적: 실제 리더보드 점수(786)와 로컬 CV 점수(2411.4)의 큰 격차 원인을 확인한다.
# 가설: 로컬 CV는 StratifiedKFold(shuffle=True)로 2019~2024를 무작위로 섞어
# 평가하지만, 실제 평가 데이터(test.csv)는 season=2025로 완전히 미래 시즌이다.
# 무작위 K-fold는 "같은 시대 분포 안에서의 보간"만 검증할 뿐 "미래 시즌으로의
# 외삽/일반화"는 전혀 검증하지 못한다 -> 이걸 시간 기반 홀드아웃으로 재현해서
# 진짜 격차가 여기서 나오는지 확인한다.
#
# 설계 (walk-forward 1-split):
#   fit period   = season 2019~2023 (실제 파이프라인의 "학습에 쓸 수 있는 전체 데이터" 역할)
#   holdout      = season 2024      (실제 평가 데이터 2025의 대역 - fit 기간에 전혀 없던 미래)
#
# fit period 안에서는 train_ensemble.py의 build_features/train_lgb를 그대로 재사용
# (내부 랜덤 5-fold는 fit 기간 내부에서만 섞이므로 여전히 leak-free).
# holdout은 build_features_infer와 동일한 "정적 스냅샷 조인" 경로를 타도록
# TARGET_COL을 제거한 뒤 build_features를 호출한다 (실제 제출 스크립트와 동일 로직).
#
# LGB만 사용한다(MLP는 GPU epoch 학습이 오래 걸려 1차 진단 목적엔 과함 -
# LGB가 단독 최고 성능 모델이라 격차 유무를 확인하는 데는 충분).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
ID_COL = te.ID_COL

BASELINE_BRIER = 0.249446  # README 기준 전체 데이터 baseline


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()
    print(f"fit(2019-2023)={len(fit_df)}  holdout(2024)={len(holdout_df)}")
    print(f"fit success_rate={fit_df[TARGET_COL].mean():.4f}  "
          f"holdout success_rate={holdout_df[TARGET_COL].mean():.4f}")

    print("\nBuild snapshot tables from fit period only (실제 제출 시나리오와 동일)...")
    snapshot_tables = te.build_train_snapshot_tables(fit_df)

    print("Build fit features (as-of cumulative, fit 기간 내부만 사용)...")
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    print("Build holdout features (정적 스냅샷 조인, script.py 추론 경로와 동일)...")
    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]

    print(f"\nn_features={len(feat_cols)}  n_fit={len(X_fit)}  n_holdout={len(X_holdout)}")

    print("\nTrain LightGBM on fit period (5-fold, 랜덤 -- fit 기간 내부에서만 섞임)...")
    lgb_models, fit_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features)

    fit_oof_brier = brier_score_loss(y_fit, fit_oof)
    print(f"\n[fit period 내부 OOF] brier={fit_oof_brier:.5f}  score={score(fit_oof_brier):.1f}"
          f"   (<- 지금까지 report된 2411.4와 같은 성격의 '무작위 K-fold' 숫자)")

    print("\nIsotonic 보정 (fit OOF로 학습, 지금까지와 동일한 절차)...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(fit_oof, y_fit)
    fit_oof_cal = iso.predict(fit_oof)
    fit_oof_cal_brier = brier_score_loss(y_fit, fit_oof_cal)
    print(f"[fit period 내부 OOF, 보정 후] brier={fit_oof_cal_brier:.5f}  score={score(fit_oof_cal_brier):.1f}")

    print("\nHoldout(2024, 완전히 못 본 미래 시즌) 예측 - fit fold 모델 평균...")
    holdout_pred_raw = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)
    holdout_raw_brier = brier_score_loss(y_holdout, holdout_pred_raw)
    print(f"[holdout 2024, 보정 전] brier={holdout_raw_brier:.5f}  score={score(holdout_raw_brier):.1f}")

    holdout_pred_cal = iso.predict(holdout_pred_raw)
    holdout_cal_brier = brier_score_loss(y_holdout, holdout_pred_cal)
    print(f"[holdout 2024, fit-OOF 보정기 적용] brier={holdout_cal_brier:.5f}  score={score(holdout_cal_brier):.1f}")

    print("\n=== 요약 ===")
    print(f"{'구분':35s} {'brier':>10s} {'score':>10s}")
    print(f"{'fit 내부 랜덤 K-fold (보정 후)':35s} {fit_oof_cal_brier:10.5f} {score(fit_oof_cal_brier):10.1f}")
    print(f"{'2024 홀드아웃 (보정 전)':35s} {holdout_raw_brier:10.5f} {score(holdout_raw_brier):10.1f}")
    print(f"{'2024 홀드아웃 (fit 보정기 적용)':35s} {holdout_cal_brier:10.5f} {score(holdout_cal_brier):10.1f}")


if __name__ == "__main__":
    main()
