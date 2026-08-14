# grid_search_lgb.py
# ------------------------------------------------------------
# LightGBM 하이퍼파라미터 2단계 순차 그리드서치
#   1단계: 핵심 구조 파라미터(learning_rate, num_leaves, min_data_in_leaf)
#          3값씩 27조합, 2-fold, 전체 데이터
#   2단계: 정규화 파라미터(feature_fraction, bagging_fraction, lambda_l2)
#          1단계 최적값을 고정하고 3값씩 27조합, 2-fold, 전체 데이터
#   최종: 전체 조합 중 최적값으로 5-fold 재검증 (기존 기본값과 비교)
# ------------------------------------------------------------
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_lgb

DATA_DIR = "../open/data"
SEARCH_FOLDS = 2
FINAL_FOLDS = 5

STAGE1_GRID = {
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_data_in_leaf": [100, 200, 400],
}
STAGE2_GRID = {
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [1.0, 5.0, 10.0],
}
DEFAULT_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 5.0,
}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}


def clean_params(d):
    """DataFrame.iloc[0]로 행을 뽑으면 정수 컬럼도 float로 섞여 나와서
    (예: num_leaves가 127.0) LightGBM이 타입 에러를 낸다. 원래 타입으로 복원."""
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
    results_df = pd.DataFrame(results).sort_values("brier")
    return results_df


def main():
    print("데이터 로드 및 피처 생성...", flush=True)
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_feat, feat_cols = build_features(train, None)
    cat_features = [c for c in CAT_COLS if c in feat_cols]
    X = train_feat[feat_cols]
    y = train_feat[TARGET_COL].values
    print(f"피처 개수: {len(feat_cols)}, 행 수: {len(X)}", flush=True)

    t_start = time.time()

    # 1단계 (이미 완료된 결과가 있으면 재사용 - 재시작 시 40분 낭비 방지)
    n_expected_1 = 1
    for v in STAGE1_GRID.values():
        n_expected_1 *= len(v)
    try:
        stage1_df = pd.read_csv("grid_search_stage1.csv").sort_values("brier")
        if len(stage1_df) != n_expected_1:
            raise ValueError("incomplete stage1 csv")
        print(f"1단계: 기존 결과 재사용 ({len(stage1_df)}조합, grid_search_stage1.csv)", flush=True)
    except (FileNotFoundError, ValueError):
        stage1_df = run_grid(X, y, cat_features, {}, STAGE1_GRID, "grid_search_stage1.csv")
    best1 = clean_params(stage1_df.iloc[0][list(STAGE1_GRID.keys())].to_dict())
    print(f"\n=== 1단계 최적: {best1} (brier={stage1_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    # 2단계 (1단계 최적값 고정)
    stage2_df = run_grid(X, y, cat_features, best1, STAGE2_GRID, "grid_search_stage2.csv")
    best2 = clean_params(stage2_df.iloc[0][list(STAGE2_GRID.keys())].to_dict())
    print(f"\n=== 2단계 최적: {best2} (brier={stage2_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    best_params = dict(best1)
    best_params.update(best2)
    print(f"=== 최종 최적 조합: {best_params} ===", flush=True)

    # 최종 검증: 기본값 vs 최적값, 5-fold
    print("\n=== 최종 검증 (5-fold, 기본값 vs 최적값) ===", flush=True)
    _, oof_default = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                                params_override=DEFAULT_PARAMS, verbose_fold=True)
    brier_default = brier_score_loss(y, oof_default)
    print(f"[기본값] brier={brier_default:.6f}", flush=True)

    _, oof_best = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                             params_override=best_params, verbose_fold=True)
    brier_best = brier_score_loss(y, oof_best)
    print(f"[튜닝값] brier={brier_best:.6f}", flush=True)

    improvement = (brier_default - brier_best) / brier_default * 100
    print(f"\n개선폭: {improvement:.4f}% (양수면 개선)", flush=True)
    print(f"기본값 파라미터: {DEFAULT_PARAMS}", flush=True)
    print(f"최적 파라미터: {best_params}", flush=True)

    with open("grid_search_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "default_params": DEFAULT_PARAMS, "default_brier": brier_default,
            "best_params": best_params, "best_brier": brier_best,
            "improvement_pct": improvement,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
