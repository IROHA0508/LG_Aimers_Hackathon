# full_stack_check.py
# ------------------------------------------------------------
# LGB+RF(2way, score=2409.2)와 LGB+MLP(2way, score=2411.5)가 각각 LGB 단독보다
# 나았던 것을 확인했으니, RF와 MLP를 동시에 스태킹하면 더 개선되는지 검증.
# RF-MLP 상관관계가 낮으면(둘이 서로 다른 오차를 만들면) 추가 이득이 있을 것이고,
# 높으면(둘 다 "LGB가 못 잡는 비슷한 패턴"을 잡는 거라면) 중복이라 이득이 없을 것.
# ------------------------------------------------------------
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb, train_rf, train_mlp

DATA_DIR = "../open/data"


def score(brier, baseline):
    return max(0, 100000 * (1 - brier / baseline))


def stack(oof_dict, y, names):
    isos = {n: IsotonicRegression(out_of_bounds="clip").fit(oof_dict[n], y) for n in names}
    calibrated = {n: isos[n].predict(oof_dict[n]) for n in names}
    meta_X = np.column_stack([calibrated[n] for n in names])
    meta = LogisticRegression().fit(meta_X, y)
    pred = meta.predict_proba(meta_X)[:, 1]
    return brier_score_loss(y, pred), meta.coef_[0]


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

    print("\n=== Entity-Embedding MLP ===", flush=True)
    t0 = time.time()
    _, mlp_oof, _ = train_mlp(X, y, cat_features)
    mlp_brier = brier_score_loss(y, mlp_oof)
    print(f"[MLP] 소요시간: {time.time()-t0:.1f}초, brier={mlp_brier:.6f}, score={score(mlp_brier, baseline_brier):.1f}", flush=True)

    oof_dict = {"lgb": lgb_oof, "rf": rf_oof, "mlp": mlp_oof}
    corr_lgb_rf = np.corrcoef(lgb_oof, rf_oof)[0, 1]
    corr_lgb_mlp = np.corrcoef(lgb_oof, mlp_oof)[0, 1]
    corr_rf_mlp = np.corrcoef(rf_oof, mlp_oof)[0, 1]
    print(f"\n상관계수: lgb-rf={corr_lgb_rf:.4f}, lgb-mlp={corr_lgb_mlp:.4f}, rf-mlp={corr_rf_mlp:.4f}", flush=True)

    lgb_score = score(lgb_brier, baseline_brier)

    print("\n=== LGB+RF 2way (재확인) ===", flush=True)
    brier_rf, coef_rf = stack(oof_dict, y, ["lgb", "rf"])
    score_rf = score(brier_rf, baseline_brier)
    print(f"[LGB+RF] brier={brier_rf:.6f}, score={score_rf:.1f}, coef={coef_rf}", flush=True)

    print("\n=== LGB+MLP 2way (재확인) ===", flush=True)
    brier_mlp, coef_mlp = stack(oof_dict, y, ["lgb", "mlp"])
    score_mlp = score(brier_mlp, baseline_brier)
    print(f"[LGB+MLP] brier={brier_mlp:.6f}, score={score_mlp:.1f}, coef={coef_mlp}", flush=True)

    print("\n=== LGB+RF+MLP 3way ===", flush=True)
    brier_3way, coef_3way = stack(oof_dict, y, ["lgb", "rf", "mlp"])
    score_3way = score(brier_3way, baseline_brier)
    print(f"[LGB+RF+MLP] brier={brier_3way:.6f}, score={score_3way:.1f}, coef(lgb,rf,mlp)={coef_3way}", flush=True)

    best_2way = max(score_rf, score_mlp)
    diff_3way = score_3way - best_2way
    print(f"\n최고 2way({'RF' if score_rf>score_mlp else 'MLP'}, {best_2way:.1f}) 대비 3way 점수차: "
          f"{diff_3way:+.1f}점 (양수면 3way가 더 나음)", flush=True)

    with open("full_stack_check_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "lgb_brier": lgb_brier, "lgb_score": lgb_score,
            "rf_brier": rf_brier, "rf_score": score(rf_brier, baseline_brier),
            "mlp_brier": mlp_brier, "mlp_score": score(mlp_brier, baseline_brier),
            "corr_lgb_rf": corr_lgb_rf, "corr_lgb_mlp": corr_lgb_mlp, "corr_rf_mlp": corr_rf_mlp,
            "stack_lgb_rf_score": score_rf, "stack_lgb_mlp_score": score_mlp,
            "stack_3way_brier": brier_3way, "stack_3way_score": score_3way,
            "diff_3way_vs_best_2way": diff_3way,
        }, f, ensure_ascii=False, indent=2)

    print("\n=== 최종 요약 ===", flush=True)
    print(f"LightGBM 단독: score={lgb_score:.1f}", flush=True)
    print(f"LGB+RF 2way: score={score_rf:.1f}", flush=True)
    print(f"LGB+MLP 2way: score={score_mlp:.1f}", flush=True)
    print(f"LGB+RF+MLP 3way: score={score_3way:.1f} ({diff_3way:+.1f} vs best 2way)", flush=True)


if __name__ == "__main__":
    main()
