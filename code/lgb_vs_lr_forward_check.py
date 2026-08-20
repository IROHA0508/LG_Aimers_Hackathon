# lgb_vs_lr_forward_check.py
# ------------------------------------------------------------
# 지금까지 모델 선택(LGB가 최고 단독 성능)은 전부 랜덤 K-fold로 결정됐다.
# 가설: LR처럼 단순한(고분산 저역량) 모델은 랜덤 fold 기준 단독 성능은
# LGB보다 한참 낮지만(README: LGB 2397.8 vs LR 1393.7), forward-time
# holdout에서는 fit-holdout 격차가 더 작아서 상대적으로 덜 무너질 수 있다
# (복잡한 트리 모델일수록 in-distribution 패턴에 더 강하게 과최적화할
# 여지가 크므로).
#
# 같은 cross-regime harness(2019-2023 학습 -> 2024 홀드아웃)로 LGB와 LR을
# 나란히 비교한다. LGB 쪽 baseline 숫자는 이미 확인됨(fit_oof_cal=2659.5,
# holdout_cal=460.2, time_holdout_check.py) - 여기서는 재확인 겸 LR을 같은
# 파이프라인에서 추가로 학습해 "격차의 상대적 크기"를 직접 비교한다.
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


def eval_model(name, oof, y_fit, holdout_pred_raw, y_holdout):
    fit_oof_brier = brier_score_loss(y_fit, oof)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_fit)
    fit_oof_cal_brier = brier_score_loss(y_fit, iso.predict(oof))

    holdout_brier = brier_score_loss(y_holdout, holdout_pred_raw)
    holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred_raw))

    fit_score = score(fit_oof_cal_brier)
    holdout_score = score(holdout_cal_brier)
    gap_ratio = fit_score / holdout_score if holdout_score > 0 else float("inf")

    print(f"\n[{name}]")
    print(f"  fit OOF(raw) brier={fit_oof_brier:.5f}  fit OOF(calibrated) score={fit_score:.1f}")
    print(f"  holdout(raw) brier={holdout_brier:.5f} score={score(holdout_brier):.1f}")
    print(f"  holdout(calibrated) brier={holdout_cal_brier:.5f} score={holdout_score:.1f}")
    print(f"  fit/holdout score ratio (붕괴 정도, 낮을수록 forward에 강함) = {gap_ratio:.2f}x")
    return dict(name=name, fit_score=fit_score, holdout_score=holdout_score, gap_ratio=gap_ratio)


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

    print("\n=== Train LightGBM (baseline, tuned params) ===")
    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    lgb_holdout_pred = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)
    results.append(eval_model("LightGBM (tuned)", lgb_oof, y_fit, lgb_holdout_pred, y_holdout))

    print("\n=== Train LogisticRegression ===")
    lr_models, lr_oof = te.train_lr(X_fit, y_fit, cat_features)
    lr_holdout_pred = np.mean(
        [pipe.predict_proba(X_holdout)[:, 1] for pipe in lr_models], axis=0)
    results.append(eval_model("LogisticRegression", lr_oof, y_fit, lr_holdout_pred, y_holdout))

    print("\n\n=== SUMMARY ===")
    print(f"{'model':22s} {'fit_score':>12s} {'holdout_score':>15s} {'gap_ratio':>12s}")
    for r in results:
        print(f"{r['name']:22s} {r['fit_score']:12.1f} {r['holdout_score']:15.1f} {r['gap_ratio']:12.2f}x")


if __name__ == "__main__":
    main()
