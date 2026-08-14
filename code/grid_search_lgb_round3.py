# grid_search_lgb_round3.py
# ------------------------------------------------------------
# 1차/2차 그리드서치 결과, 아래 3개 파라미터가 계속 탐색 범위 경계에 몰림:
#   learning_rate=0.01(최소), num_leaves=255(최대), min_data_in_leaf=800(최대)
# 반면 feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=10.0은 2차에서
# 중간값으로 안정됐으므로 3차에서는 재탐색하지 않고 고정한다.
#
# 3차는 경계였던 3개만 더 확장해서 단일 단계(27조합)로 탐색한다.
# learning_rate가 더 작아지면 수렴에 더 많은 라운드가 필요할 수 있어
# num_boost_round 상한과 early_stopping patience를 늘려서 조기 종료로
# 인한 불공정한 비교(저학습률 조합이 덜 수렴된 채로 저평가되는 것)를 방지한다.
#
# 실행 시간이 1~2.5시간대로 걸릴 수 있어(정확한 시간은 저학습률 수렴
# 속도에 달려있어 사전에 확신하기 어려움), VS Code를 끄고 나중에 실행해도
# 되도록 매 조합마다 CSV에 즉시 저장하고, 중간에 끊겨도 재실행 시
# 이미 완료된 조합은 건너뛰도록 재개(resume) 기능을 넣었다.
#
# 실행: python grid_search_lgb_round3.py
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
SEARCH_EARLY_STOPPING = 300   # 1차/2차(200)보다 여유를 둠 - 저학습률 조합이 손해보지 않도록
SEARCH_NUM_BOOST_ROUND = 8000  # 1차/2차(4000)의 2배 - learning_rate=0.005까지 내려가므로

STAGE1_GRID = {
    "learning_rate": [0.005, 0.0075, 0.01],
    "num_leaves": [255, 320, 400],
    "min_data_in_leaf": [800, 1200, 1600],
}
ROUND2_BEST_PARAMS = {
    "learning_rate": 0.01, "num_leaves": 255, "min_data_in_leaf": 800,
    "bagging_fraction": 0.8, "feature_fraction": 0.5, "lambda_l2": 10.0,
}
# 3차에서 고정할 값 (2차에서 이미 중간값으로 안정된 것들)
FIXED_PARAMS = {
    "feature_fraction": ROUND2_BEST_PARAMS["feature_fraction"],
    "bagging_fraction": ROUND2_BEST_PARAMS["bagging_fraction"],
    "lambda_l2": ROUND2_BEST_PARAMS["lambda_l2"],
}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}
STAGE1_LOG = "grid_search_r3_stage1.csv"


def clean_params(d):
    return {k: (int(v) if k in INT_PARAMS else float(v)) for k, v in d.items()}


def run_grid(X, y, cat_features, base_params, grid, log_path):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    # 재개(resume): 이미 끝난 조합은 다시 안 돌림 (VS Code 종료 등으로 중간에
    # 끊겼다가 재실행하는 상황을 대비)
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
        _, oof = train_lgb(
            X, y, X, cat_features,
            n_folds=SEARCH_FOLDS, params_override=override,
            early_stopping_rounds=SEARCH_EARLY_STOPPING,
            num_boost_round=SEARCH_NUM_BOOST_ROUND,
            verbose_fold=False,
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

    stage1_df = run_grid(X, y, cat_features, FIXED_PARAMS, STAGE1_GRID, STAGE1_LOG)
    best1 = clean_params(stage1_df.iloc[0][list(STAGE1_GRID.keys())].to_dict())
    print(f"\n=== 3차 최적: {best1} (brier={stage1_df.iloc[0]['brier']:.6f}) ===\n", flush=True)

    best_params = dict(FIXED_PARAMS)
    best_params.update(best1)
    print(f"=== 3차 최종 조합: {best_params} ===", flush=True)

    # 최종 검증: 2차 최적값 vs 3차 확장 최적값, 5-fold
    # (학습률이 더 낮아졌으니 최종 검증도 늘어난 num_boost_round/patience로 공정하게 비교)
    print("\n=== 최종 검증 (5-fold, 2차 최적값 vs 3차 확장 최적값) ===", flush=True)
    _, oof_r2 = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                           params_override=ROUND2_BEST_PARAMS,
                           early_stopping_rounds=SEARCH_EARLY_STOPPING,
                           num_boost_round=SEARCH_NUM_BOOST_ROUND,
                           verbose_fold=True)
    brier_r2 = brier_score_loss(y, oof_r2)
    print(f"[2차 최적값] brier={brier_r2:.6f}", flush=True)

    _, oof_r3 = train_lgb(X, y, X, cat_features, n_folds=FINAL_FOLDS,
                           params_override=best_params,
                           early_stopping_rounds=SEARCH_EARLY_STOPPING,
                           num_boost_round=SEARCH_NUM_BOOST_ROUND,
                           verbose_fold=True)
    brier_r3 = brier_score_loss(y, oof_r3)
    print(f"[3차 확장 최적값] brier={brier_r3:.6f}", flush=True)

    improvement = (brier_r2 - brier_r3) / brier_r2 * 100
    print(f"\n2차 대비 개선폭: {improvement:.4f}% (양수면 개선)", flush=True)
    print(f"2차 파라미터: {ROUND2_BEST_PARAMS}", flush=True)
    print(f"3차 파라미터: {best_params}", flush=True)

    with open("grid_search_round3_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "round2_params": ROUND2_BEST_PARAMS, "round2_brier": brier_r2,
            "round3_params": best_params, "round3_brier": brier_r3,
            "improvement_pct_vs_round2": improvement,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
