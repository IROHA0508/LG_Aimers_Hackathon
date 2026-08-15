# final_check.py
# ------------------------------------------------------------
# 지금까지 반영된 모든 변경사항(82피처 + 튜닝된 LGB/XGB 기본값 + 기본 CAT)으로
# 3모델(LGB/XGB/CAT) 스태킹까지 전체 파이프라인을 처음부터 다시 돌려서
# 최종 점수를 확인한다. randomforest.ipynb의 전체 데이터 섹션과 동일한 로직.
# ------------------------------------------------------------
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb, train_xgb, train_cat

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

    results = {}

    print("\n=== LightGBM (튜닝됨) ===", flush=True)
    t0 = time.time()
    _, lgb_oof = train_lgb(X, y, X, cat_features)
    lgb_brier = brier_score_loss(y, lgb_oof)
    print(f"[LGB] 소요시간: {time.time()-t0:.1f}초, brier={lgb_brier:.6f}, score={score(lgb_brier, baseline_brier):.1f}", flush=True)
    results["lgb_brier"] = lgb_brier

    print("\n=== XGBoost (튜닝됨) ===", flush=True)
    t0 = time.time()
    _, xgb_oof = train_xgb(X, y, cat_features)
    xgb_brier = brier_score_loss(y, xgb_oof)
    print(f"[XGB] 소요시간: {time.time()-t0:.1f}초, brier={xgb_brier:.6f}, score={score(xgb_brier, baseline_brier):.1f}", flush=True)
    results["xgb_brier"] = xgb_brier

    print("\n=== CatBoost (기본값) ===", flush=True)
    t0 = time.time()
    _, cat_oof = train_cat(X, y, cat_features)
    cat_brier = brier_score_loss(y, cat_oof)
    print(f"[CAT] 소요시간: {time.time()-t0:.1f}초, brier={cat_brier:.6f}, score={score(cat_brier, baseline_brier):.1f}", flush=True)
    results["cat_brier"] = cat_brier

    print("\n=== 3모델 로지스틱 스태킹 ===", flush=True)
    iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y)
    iso_xgb = IsotonicRegression(out_of_bounds="clip").fit(xgb_oof, y)
    iso_cat = IsotonicRegression(out_of_bounds="clip").fit(cat_oof, y)
    lgb_c, xgb_c, cat_c = iso_lgb.predict(lgb_oof), iso_xgb.predict(xgb_oof), iso_cat.predict(cat_oof)

    meta_X = np.column_stack([lgb_c, xgb_c, cat_c])
    meta = LogisticRegression().fit(meta_X, y)
    stack_oof = meta.predict_proba(meta_X)[:, 1]
    stack_brier = brier_score_loss(y, stack_oof)
    stack_score = score(stack_brier, baseline_brier)
    print(f"[3모델 스태킹] brier={stack_brier:.6f}, score={stack_score:.1f}", flush=True)
    print(f"메타 계수: lgb={meta.coef_[0][0]:.3f}, xgb={meta.coef_[0][1]:.3f}, cat={meta.coef_[0][2]:.3f}", flush=True)

    results["stack_brier"] = stack_brier
    results["stack_score"] = stack_score
    results["baseline_brier"] = baseline_brier
    results["n_features"] = len(feat_cols)

    with open("final_check_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 최종 요약 ===", flush=True)
    for name, b in [("LightGBM", lgb_brier), ("XGBoost", xgb_brier), ("CatBoost", cat_brier), ("3모델 스태킹", stack_brier)]:
        print(f"{name}: brier={b:.6f}, score={score(b, baseline_brier):.1f}", flush=True)


if __name__ == "__main__":
    main()
