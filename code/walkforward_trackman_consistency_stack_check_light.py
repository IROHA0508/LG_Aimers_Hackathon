# walkforward_trackman_consistency_stack_check_light.py
# ------------------------------------------------------------
# walkforward_trackman_consistency_stack_check.py(5개 시즌 x 2 설정 = 10회
# LGB+RF+MLP 풀 학습)를 로컬에서 돌리다 메모리 부족(총 15.6GB 중 여유 2.7GB)으로
# RF의 n_jobs=-1이 조인트 프로세스를 과도하게 늘려 스와핑에 빠져 4시간+ 무응답 -> 강제
# 종료했다. 이 버전은 그 문제를 피하려고:
#   1) 가장 배포 상황과 가까운(fit 데이터가 가장 큰) 최근 폴드만 우선 확인
#      (target_season=2024, fit=2019-2023 약 122만행)
#   2) LOKY_MAX_CPU_COUNT로 joblib/RF 워커 프로세스 수를 4개로 제한해 메모리
#      배수 폭발을 방지
#   3) 매 스텝 print(flush=True) + gc.collect()로 진행상황을 즉시 볼 수 있게 함
# ------------------------------------------------------------
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import gc
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from trackman_pitcher_lib import build_pitcher_id_map, build_trackman_consistency_prior
from walkforward_harness import score, DATA_DIR

TARGET_COL = te.TARGET_COL
FEATURE_KWARGS = dict(use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False)
TARGET_SEASONS = [2024]  # 필요시 [2022, 2024]로 확장


def log(msg):
    print(msg, flush=True)


def prep_fold(train, target_season, tm=None):
    fit_df = train[train["season"] < target_season].copy()
    holdout_df = train[train["season"] == target_season].copy()

    if tm is not None:
        cutoff_season = target_season - 1
        id_map = build_pitcher_id_map(fit_df, tm, min_season=fit_df["season"].min(), cutoff_season=cutoff_season)
        prior = build_trackman_consistency_prior(tm, cutoff_season, id_map)
        fit_df = fit_df.merge(prior, on="pitcher_id", how="left")
        holdout_df = holdout_df.merge(prior, on="pitcher_id", how="left")

    snapshot_tables = te.build_train_snapshot_tables(fit_df)
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None, **FEATURE_KWARGS)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories, **FEATURE_KWARGS)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]
    return X_fit, y_fit, X_holdout, y_holdout, cat_features


def fit_stack(oof_dict, y_fit, pred_dict):
    calibrators = {}
    calibrated_oof = {}
    for name, oof in oof_dict.items():
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)
        calibrators[name] = iso
        calibrated_oof[name] = iso.predict(oof)

    names = list(oof_dict.keys())
    meta_X = np.column_stack([calibrated_oof[n] for n in names])
    meta = LogisticRegression()
    meta.fit(meta_X, y_fit)
    stack_pred_fit = meta.predict_proba(meta_X)[:, 1]

    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(stack_pred_fit, y_fit)

    calibrated_pred = {n: calibrators[n].predict(pred_dict[n]) for n in names}
    meta_X_h = np.column_stack([calibrated_pred[n] for n in names])
    stack_pred_h = meta.predict_proba(meta_X_h)[:, 1]
    return final_iso.predict(stack_pred_h)


def run_fold_3way(train, target_season, tm=None, tag=""):
    log(f"  [{tag}] prep_fold...")
    X_fit, y_fit, X_holdout, y_holdout, cat_features = prep_fold(train, target_season, tm=tm)
    log(f"  [{tag}] X_fit={X_fit.shape} X_holdout={X_holdout.shape}")

    log(f"  [{tag}] train_lgb...")
    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    lgb_pred_h = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)
    log(f"  [{tag}] train_lgb done")

    log(f"  [{tag}] train_mlp...")
    mlp_state_dicts, mlp_oof, mlp_preproc = te.train_mlp(X_fit, y_fit, cat_features, verbose_fold=False)
    mlp_pred_h = te.predict_mlp(mlp_state_dicts, mlp_preproc, X_holdout)
    log(f"  [{tag}] train_mlp done")

    log(f"  [{tag}] train_rf (n_jobs capped via LOKY_MAX_CPU_COUNT=4)...")
    rf_models, rf_oof = te.train_rf(X_fit, y_fit, cat_features)
    rf_pred_h = np.mean([pipe.predict_proba(X_holdout)[:, 1] for pipe in rf_models], axis=0)
    log(f"  [{tag}] train_rf done")

    pred = fit_stack({"lgb": lgb_oof, "rf": rf_oof, "mlp": mlp_oof}, y_fit,
                      {"lgb": lgb_pred_h, "rf": rf_pred_h, "mlp": mlp_pred_h})
    s = score(brier_score_loss(y_holdout, pred))

    del X_fit, X_holdout, lgb_models, mlp_state_dicts, rf_models
    gc.collect()
    return s


def main():
    log("Load data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    tm["team_merged"] = tm["pitcher_team"].replace({"SSG_LAN": "SK_WYV"})

    for target_season in TARGET_SEASONS:
        log(f"=== target_season={target_season} ===")
        base = run_fold_3way(train, target_season, tm=None, tag=f"{target_season}-base")
        gc.collect()
        with_tm = run_fold_3way(train, target_season, tm=tm, tag=f"{target_season}-trackman")
        delta = with_tm - base
        log(f"predict {target_season}: LGB+RF+MLP base={base:.1f}  +trackman_consistency={with_tm:.1f}  delta={delta:+.1f}")


if __name__ == "__main__":
    main()
