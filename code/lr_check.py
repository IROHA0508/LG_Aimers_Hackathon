# lr_check.py
# ------------------------------------------------------------
# LGB+RF 2모델 스태킹(1차 결과, score=2409.2)에 LogisticRegression(train_lr)을
# 세 번째 축으로 추가하면 더 개선되는지 검증.
# LR은 선형·가법 모델이라 트리(LGB)/배깅(RF)과는 또 다른 종류의 오차를 만들
# 가능성이 있음 - RF가 낮은 상관계수(0.939)로 다양성 이득을 준 것과 같은 논리를
# 선형모델에도 적용해보는 실험.
# ------------------------------------------------------------
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb, train_rf, train_lr

DATA_DIR = "../open/data"


def score(brier, baseline):
    return max(0, 100000 * (1 - brier / baseline))


def stack(oof_dict, y, names):
    isos = {n: IsotonicRegression(out_of_bounds="clip").fit(oof_dict[n], y) for n in names}
    calibrated = {n: isos[n].predict(oof_dict[n]) for n in names}
    meta_X = np.column_stack([calibrated[n] for n in names])
    meta = LogisticRegression().fit(meta_X, y)
    pred = meta.predict_proba(meta_X)[:, 1]
    brier = brier_score_loss(y, pred)
    return brier, meta.coef_[0]


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

    print("\n=== LogisticRegression (기본값) ===", flush=True)
    t0 = time.time()
    _, lr_oof = train_lr(X, y, cat_features)
    lr_brier = brier_score_loss(y, lr_oof)
    print(f"[LR] 소요시간: {time.time()-t0:.1f}초, brier={lr_brier:.6f}, score={score(lr_brier, baseline_brier):.1f}", flush=True)

    oof_dict = {"lgb": lgb_oof, "rf": rf_oof, "lr": lr_oof}
    corr_lgb_rf = np.corrcoef(lgb_oof, rf_oof)[0, 1]
    corr_lgb_lr = np.corrcoef(lgb_oof, lr_oof)[0, 1]
    corr_rf_lr = np.corrcoef(rf_oof, lr_oof)[0, 1]
    print(f"\n상관계수: lgb-rf={corr_lgb_rf:.4f}, lgb-lr={corr_lgb_lr:.4f}, rf-lr={corr_rf_lr:.4f}", flush=True)

    print("\n=== LGB+RF 2모델 스태킹 (기존 1차 결과 재확인) ===", flush=True)
    brier_2way, coef_2way = stack(oof_dict, y, ["lgb", "rf"])
    score_2way = score(brier_2way, baseline_brier)
    print(f"[LGB+RF] brier={brier_2way:.6f}, score={score_2way:.1f}, coef(lgb,rf)={coef_2way}", flush=True)

    print("\n=== LGB+RF+LR 3모델 스태킹 ===", flush=True)
    brier_3way, coef_3way = stack(oof_dict, y, ["lgb", "rf", "lr"])
    score_3way = score(brier_3way, baseline_brier)
    print(f"[LGB+RF+LR] brier={brier_3way:.6f}, score={score_3way:.1f}, coef(lgb,rf,lr)={coef_3way}", flush=True)

    lgb_score = score(lgb_brier, baseline_brier)
    diff_2way = score_2way - lgb_score
    diff_3way = score_3way - score_2way
    print(f"\nLGB 단독 대비 2way 점수차: {diff_2way:+.1f}점", flush=True)
    print(f"2way 대비 3way(LR 추가) 점수차: {diff_3way:+.1f}점 (양수면 LR 추가가 도움됨)", flush=True)

    with open("lr_check_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "lgb_brier": lgb_brier, "lgb_score": lgb_score,
            "rf_brier": rf_brier, "rf_score": score(rf_brier, baseline_brier),
            "lr_brier": lr_brier, "lr_score": score(lr_brier, baseline_brier),
            "corr_lgb_rf": corr_lgb_rf, "corr_lgb_lr": corr_lgb_lr, "corr_rf_lr": corr_rf_lr,
            "stack_2way_brier": brier_2way, "stack_2way_score": score_2way,
            "stack_3way_brier": brier_3way, "stack_3way_score": score_3way,
            "diff_2way_vs_lgb": diff_2way,
            "diff_3way_vs_2way": diff_3way,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== 최종 요약 ===", flush=True)
    print(f"LightGBM 단독: score={lgb_score:.1f}", flush=True)
    print(f"LGB+RF 2way: score={score_2way:.1f} ({diff_2way:+.1f})", flush=True)
    print(f"LGB+RF+LR 3way: score={score_3way:.1f} ({diff_3way:+.1f} vs 2way)", flush=True)


if __name__ == "__main__":
    main()
