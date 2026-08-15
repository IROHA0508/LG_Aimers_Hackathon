# mlp_check.py
# ------------------------------------------------------------
# Entity-Embedding MLP(딥러닝 후보)가 LGB 단독/LGB+RF 1차 결과 대비
# 스태킹 다양성을 더해주는지 검증. RF/LR과 같은 논리 - 단독 성능이 LGB보다
# 낮더라도 오차 상관관계가 낮으면 스태킹에 도움이 될 수 있다는 가설.
#
# 주의: 지금 바로 실행하지 않는다. lr_check.py(LGB+RF+LR 3way) 결과를 먼저
# 분석한 뒤에 이어서 검증할 예정 - 코드만 미리 작성/저장해둔 상태.
# ------------------------------------------------------------
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb, train_mlp

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

    print("\n=== Entity-Embedding MLP ===", flush=True)
    t0 = time.time()
    _, mlp_oof, _ = train_mlp(X, y, cat_features)
    mlp_brier = brier_score_loss(y, mlp_oof)
    print(f"[MLP] 소요시간: {time.time()-t0:.1f}초, brier={mlp_brier:.6f}, score={score(mlp_brier, baseline_brier):.1f}", flush=True)

    corr = np.corrcoef(lgb_oof, mlp_oof)[0, 1]
    print(f"\nLGB-MLP 상관계수: {corr:.4f}", flush=True)

    print("\n=== LGB+MLP 2모델 스태킹 ===", flush=True)
    iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y)
    iso_mlp = IsotonicRegression(out_of_bounds="clip").fit(mlp_oof, y)
    lgb_c, mlp_c = iso_lgb.predict(lgb_oof), iso_mlp.predict(mlp_oof)

    meta_X = np.column_stack([lgb_c, mlp_c])
    meta = LogisticRegression().fit(meta_X, y)
    stack_oof = meta.predict_proba(meta_X)[:, 1]
    stack_brier = brier_score_loss(y, stack_oof)
    stack_score = score(stack_brier, baseline_brier)
    print(f"[LGB+MLP 스태킹] brier={stack_brier:.6f}, score={stack_score:.1f}", flush=True)
    print(f"메타 계수: lgb={meta.coef_[0][0]:.3f}, mlp={meta.coef_[0][1]:.3f}", flush=True)

    lgb_score = score(lgb_brier, baseline_brier)
    diff = stack_score - lgb_score
    print(f"\nLGB 단독 대비 스태킹 점수차: {diff:+.1f}점 (양수면 MLP 추가가 도움됨)", flush=True)

    with open("mlp_check_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "lgb_brier": lgb_brier, "lgb_score": lgb_score,
            "mlp_brier": mlp_brier, "mlp_score": score(mlp_brier, baseline_brier),
            "lgb_mlp_corr": corr,
            "stack_brier": stack_brier, "stack_score": stack_score,
            "diff_vs_lgb_alone": diff,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== 최종 요약 ===", flush=True)
    print(f"LightGBM 단독: score={lgb_score:.1f}", flush=True)
    print(f"LGB+MLP 스태킹: score={stack_score:.1f} ({diff:+.1f})", flush=True)


if __name__ == "__main__":
    main()
