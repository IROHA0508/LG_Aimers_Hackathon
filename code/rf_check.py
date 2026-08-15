# rf_check.py
# ------------------------------------------------------------
# 튜닝된 LGB 단독 vs (튜닝 안 된 RF + 튜닝된 LGB) 2모델 스태킹을 비교.
# LGB가 XGB/CAT보다 훨씬 앞서면서 3모델 스태킹조차 LGB 단독보다 못했던 것과
# 같은 이유로, RF(더 약한 모델)를 섞어도 손해일 가능성이 높다는 가설을 검증.
# ------------------------------------------------------------
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb, train_rf

DATA_DIR = "../open/data"


def score(brier, baseline):
    return max(0, 100000 * (1 - brier / baseline))


def main():
    print("데이터 로드 및 피처 생성...", flush=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_feat, feat_cols = build_features(train, None)
    cat_features = [c for c in CAT_COLS if c in feat_cols]
    X = train_feat[feat_cols]
    y = train_feat[TARGET_COL].values
    r = y.mean()
    baseline_brier = r * (1 - r)
    print(f"피처 개수: {len(feat_cols)}, 행 수: {len(X)}, 기준 Brier: {baseline_brier:.5f}", flush=True)

    print("\n=== LightGBM (튜닝됨) ===", flush=True)
    t0 = time.time()
    _, lgb_oof = train_lgb(X, y, X, cat_features)
    lgb_brier = brier_score_loss(y, lgb_oof)
    print(f"[LGB] 소요시간: {time.time()-t0:.1f}초, brier={lgb_brier:.6f}, score={score(lgb_brier, baseline_brier):.1f}", flush=True)

    print("\n=== RandomForest (기본값) ===", flush=True)
    t0 = time.time()
    _, rf_oof = train_rf(X, y, cat_features)
    rf_brier = brier_score_loss(y, rf_oof)
    print(f"[RF] 소요시간: {time.time()-t0:.1f}초, brier={rf_brier:.6f}, score={score(rf_brier, baseline_brier):.1f}", flush=True)

    corr = np.corrcoef(lgb_oof, rf_oof)[0, 1]
    print(f"\nLGB-RF 상관계수: {corr:.4f}", flush=True)

    print("\n=== LGB+RF 2모델 스태킹 ===", flush=True)
    iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y)
    iso_rf = IsotonicRegression(out_of_bounds="clip").fit(rf_oof, y)
    lgb_c, rf_c = iso_lgb.predict(lgb_oof), iso_rf.predict(rf_oof)

    meta_X = np.column_stack([lgb_c, rf_c])
    meta = LogisticRegression().fit(meta_X, y)
    stack_oof = meta.predict_proba(meta_X)[:, 1]
    stack_brier = brier_score_loss(y, stack_oof)
    stack_score = score(stack_brier, baseline_brier)
    print(f"[LGB+RF 스태킹] brier={stack_brier:.6f}, score={stack_score:.1f}", flush=True)
    print(f"메타 계수: lgb={meta.coef_[0][0]:.3f}, rf={meta.coef_[0][1]:.3f}", flush=True)

    lgb_score = score(lgb_brier, baseline_brier)
    diff = stack_score - lgb_score
    print(f"\nLGB 단독 대비 스태킹 점수차: {diff:+.1f}점 (양수면 RF 추가가 도움됨)", flush=True)

    with open("rf_check_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "lgb_brier": lgb_brier, "lgb_score": lgb_score,
            "rf_brier": rf_brier, "rf_score": score(rf_brier, baseline_brier),
            "lgb_rf_corr": corr,
            "stack_brier": stack_brier, "stack_score": stack_score,
            "diff_vs_lgb_alone": diff,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== 최종 요약 ===", flush=True)
    print(f"LightGBM 단독: score={lgb_score:.1f}", flush=True)
    print(f"LGB+RF 스태킹: score={stack_score:.1f} ({diff:+.1f})", flush=True)


if __name__ == "__main__":
    main()
