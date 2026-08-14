# grid_search_lgb_round2.py
# ------------------------------------------------------------
# 1차 그리드서치(grid_search_lgb.py) 결과, 최적값이 경계에 몰려있었음:
#   learning_rate=0.02(최소), num_leaves=127(최대), min_data_in_leaf=400(최대),
#   feature_fraction=0.6(최소), lambda_l2=10.0(최대)  <- bagging_fraction=0.8만 중간값(안정)
# 그래서 경계 방향으로 범위를 확장해서 2차 탐색.
#   1단계: learning_rate/num_leaves/min_data_in_leaf 확장 (27조합)
#   2단계: feature_fraction/lambda_l2 확장 (9조합, bagging_fraction=0.8 고정 유지)
#   최종: 1차 최적값 vs 2차 최적값을 5-fold로 비교
# ------------------------------------------------------------
import itertools
import json
import time

import pandas as pd
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb

DATA_DIR = "../open/data"
SEARCH_FOLDS = 2
FINAL_FOLDS = 5

STAGE1_GRID = {
    "learning_rate": [0.01, 0.015, 0.02],
    "num_leaves": [127, 191, 255],
    "min_data_in_leaf": [400, 600, 800],
}
STAGE2_GRID = {
    "feature_fraction": [0.4, 0.5, 0.6],
    "lambda_l2": [10.0, 15.0, 20.0],
}
ROUND1_BEST_PARAMS = {
    "learning_rate": 0.02, "num_leaves": 127, "min_data_in_leaf": 400,
    "feature_fraction": 0.6, "bagging_fraction": 0.8, "lambda_l2": 10.0,
}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}


def clean_params(d):
    return {k: (int(v) if k in INT_PARAMS else float(v)) for k, v in d.items()}


def run_grid(X, y, cat_features, base_params, grid, log_path):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    results = []
    print(f"=== 그리드 {len(combos)}조합 시작 (base={base_params}) ===", flush=True)
    for i, combo in enumerate(combos):
        override = dict(base_params)
        override.update(dict(zip(keys, combo)))
        t0 = time.time()
        _, oof = train_lgb(
            X, y, X, cat_features,
            n_folds=SEARCH_FOLDS, params_override=override,
            early_stopping_rounds=200, verbose_fold=False,
        )
        brier = brier_score_loss(y, oof)
        elapsed = time.time() - t0
        row = dict(override)
        row["brier"] = brier
        row["elapsed_sec"] = elapsed
        results.append(row)
        print(f"[{i+1}/{len(combos)}] {dict(zip(keys, combo))} "
              f"-> brier={brier:.6f} ({elapsed:.1f}s)", flush=True)
        pd.DataFrame(results).to_csv(log_path, index=False)
    return pd.DataFrame(results).sort_values("brier")


def main():
    print("데이터 로드 및 피처 생성...", flush=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_feat, feat_cols = build_features(train, None)
    cat_features = [c for c in CAT_COLS if c in feat_cols]
    X = train_feat[feat_cols]
    y = train_feat[TARGET_COL].values
    print(f"피처 개수: {len(feat_cols)}, 행 수: {len(X)}", flush=True)

    t_start = time.time()

    # 1단계: 구조 파라미터 확장 탐색 (bagging_fraction=0.8은 base에 포함해 고정)
    base1 = {"bagging_fraction": ROUND1_BEST_PARAMS["bagging_fraction"]}
    stage1_df = run_grid(X, y, cat_features, base1, STAGE1_GRID, "grid_search_r2_stage1.csv")
    best1 = clean_params(stage1_df.iloc[0][list(STAGE1_GRID.keys())].to_dict())
    print(f"\n=== 1단계(확장) 최적: {best1} (brier={stage1_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    # 2단계: 정규화 파라미터 확장 탐색 (1단계 최적값 + bagging_fraction 고정)
    base2 = dict(best1)
    base2["bagging_fraction"] = ROUND1_BEST_PARAMS["bagging_fraction"]
    stage2_df = run_grid(X, y, cat_features, base2, STAGE2_GRID, "grid_search_r2_stage2.csv")
    best2 = clean_params(stage2_df.iloc[0][list(STAGE2_GRID.keys())].to_dict())
    print(f"\n=== 2단계(확장) 최적: {best2} (brier={stage2_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    best_params = dict(best1)
    best_params["bagging_fraction"] = ROUND1_BEST_PARAMS["bagging_fraction"]
    best_params.update(best2)
    print(f"=== 2차 최종 최적 조합: {best_params} ===", flush=True)

    # 최종 검증: 1차 최적값 vs 2차 최적값, 5-fold
    print("\n=== 최종 검증 (5-fold, 1차 최적값 vs 2차 확장 최적값) ===", flush=True)
    _, oof_r1 = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                           params_override=ROUND1_BEST_PARAMS, verbose_fold=True)
    brier_r1 = brier_score_loss(y, oof_r1)
    print(f"[1차 최적값] brier={brier_r1:.6f}", flush=True)

    _, oof_r2 = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                           params_override=best_params, verbose_fold=True)
    brier_r2 = brier_score_loss(y, oof_r2)
    print(f"[2차 확장 최적값] brier={brier_r2:.6f}", flush=True)

    improvement = (brier_r1 - brier_r2) / brier_r1 * 100
    print(f"\n1차 대비 개선폭: {improvement:.4f}% (양수면 개선)", flush=True)
    print(f"1차 파라미터: {ROUND1_BEST_PARAMS}", flush=True)
    print(f"2차 파라미터: {best_params}", flush=True)

    with open("grid_search_round2_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "round1_params": ROUND1_BEST_PARAMS, "round1_brier": brier_r1,
            "round2_params": best_params, "round2_brier": brier_r2,
            "improvement_pct_vs_round1": improvement,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
