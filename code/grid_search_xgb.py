# grid_search_xgb.py
# ------------------------------------------------------------
# XGBoost 하이퍼파라미터 2단계 순차 그리드서치 (grid_search_lgb.py와 동일 방식)
#   1단계: 핵심 구조 파라미터(eta, max_depth, min_child_weight)
#          3값씩 27조합, 2-fold, 전체 데이터
#   2단계: 정규화 파라미터(subsample, colsample_bytree, reg_lambda)
#          1단계 최적값을 고정하고 3값씩 27조합, 2-fold, 전체 데이터
#   최종: 기본값 vs 최적값을 5-fold로 재검증
#
# LGB 그리드서치 때 겪은 문제 반영:
# - DataFrame.iloc[0] 행 추출 시 int 컬럼이 float로 섞이는 버그 -> clean_params로 방지
# - 중간에 끊길 수 있으므로(VS Code 종료 등) 조합마다 CSV 즉시 저장 + 재개(resume) 지원
# ------------------------------------------------------------
import itertools
import json
import time

import pandas as pd
from sklearn.metrics import brier_score_loss

from train_ensemble import TARGET_COL, CAT_COLS, build_features, train_xgb

DATA_DIR = "../open/data"
SEARCH_FOLDS = 2
FINAL_FOLDS = 5

STAGE1_GRID = {
    "eta": [0.02, 0.03, 0.05],
    "max_depth": [6, 8, 10],
    "min_child_weight": [10, 20, 40],
}
STAGE2_GRID = {
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}
DEFAULT_PARAMS = {
    "eta": 0.03, "max_depth": 8, "min_child_weight": 20,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
}
INT_PARAMS = {"max_depth"}


def clean_params(d):
    return {k: (int(v) if k in INT_PARAMS else float(v)) for k, v in d.items()}


def run_grid(X, y, cat_features, base_params, grid, log_path):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    done_rows = []
    done_keys = set()
    try:
        prev_df = pd.read_csv(log_path)
        done_rows = prev_df.to_dict("records")
        for r in done_rows:
            done_keys.add(tuple(r[k] for k in keys))
        print(f"기존 결과 {len(done_rows)}개 재사용 (재개 모드)", flush=True)
    except FileNotFoundError:
        pass

    results = list(done_rows)
    print(f"=== 그리드 {len(combos)}조합 시작 (base={base_params}) ===", flush=True)
    for i, combo in enumerate(combos):
        if combo in done_keys:
            continue
        override = dict(base_params)
        override.update(dict(zip(keys, combo)))
        t0 = time.time()
        _, oof = train_xgb(
            X, y, cat_features,
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

    stage1_df = run_grid(X, y, cat_features, {}, STAGE1_GRID, "grid_search_xgb_stage1.csv")
    best1 = clean_params(stage1_df.iloc[0][list(STAGE1_GRID.keys())].to_dict())
    print(f"\n=== 1단계 최적: {best1} (brier={stage1_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    stage2_df = run_grid(X, y, cat_features, best1, STAGE2_GRID, "grid_search_xgb_stage2.csv")
    best2 = clean_params(stage2_df.iloc[0][list(STAGE2_GRID.keys())].to_dict())
    print(f"\n=== 2단계 최적: {best2} (brier={stage2_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    best_params = dict(best1)
    best_params.update(best2)
    print(f"=== 최종 최적 조합: {best_params} ===", flush=True)

    print("\n=== 최종 검증 (5-fold, 기본값 vs 최적값) ===", flush=True)
    _, oof_default = train_xgb(X, y, cat_features, n_folds=FINAL_FOLDS,
                                params_override=DEFAULT_PARAMS, verbose_fold=True)
    brier_default = brier_score_loss(y, oof_default)
    print(f"[기본값] brier={brier_default:.6f}", flush=True)

    _, oof_best = train_xgb(X, y, cat_features, n_folds=FINAL_FOLDS,
                             params_override=best_params, verbose_fold=True)
    brier_best = brier_score_loss(y, oof_best)
    print(f"[튜닝값] brier={brier_best:.6f}", flush=True)

    improvement = (brier_default - brier_best) / brier_default * 100
    print(f"\n개선폭: {improvement:.4f}% (양수면 개선)", flush=True)
    print(f"기본값 파라미터: {DEFAULT_PARAMS}", flush=True)
    print(f"최적 파라미터: {best_params}", flush=True)

    with open("grid_search_xgb_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "default_params": DEFAULT_PARAMS, "default_brier": brier_default,
            "best_params": best_params, "best_brier": brier_best,
            "improvement_pct": improvement,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
