# walkforward_trackman_check.py
# ------------------------------------------------------------
# trackman_pitcher_feature_check.py에서 확인한 +21.1점 개선(2024 폴드 단일
# 분할 기준)이 다른 연도에도 재현되는지 5-fold 확장 윈도우 전체로 재검증한다.
# walkforward_harness.py의 baseline 결과(폴드별: 0.0 / 1189.2 / 1880.8 / 0.0 /
# 460.2, 평균 706.0)와 폴드별로 직접 비교한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from trackman_pitcher_lib import attach_trackman_pitcher_features
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR

TARGET_COL = te.TARGET_COL

BASELINE_FOLD_SCORES = {2020: 0.0, 2021: 1189.2, 2022: 1880.8, 2023: 0.0, 2024: 460.2}


def run_one_fold_trackman(train, tm, target_season, n_internal_folds=5):
    fit_df = train[train["season"] < target_season].copy()
    holdout_df = train[train["season"] == target_season].copy()
    cutoff_season = target_season - 1

    fit_df, holdout_df, meta = attach_trackman_pitcher_features(fit_df, holdout_df, tm, cutoff_season)

    snapshot_tables = te.build_train_snapshot_tables(fit_df)
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables, cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]

    models, oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, n_folds=n_internal_folds, verbose_fold=False)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_fit)

    holdout_pred = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
    holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred))
    holdout_score = score(holdout_cal_brier)

    return dict(target_season=target_season, n_features=len(feat_cols), meta=meta, holdout_score=holdout_score)


def main():
    print("Load data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    tm["team_merged"] = tm["pitcher_team"].replace({"SSG_LAN": "SK_WYV"})

    results = []
    for target_season in FOLD_SEASONS:
        r = run_one_fold_trackman(train, tm, target_season)
        base = BASELINE_FOLD_SCORES[target_season]
        delta = r["holdout_score"] - base
        print(f"predict {target_season}: n_mapped={r['meta']['n_mapped']} "
              f"match_rate(fit/holdout)={r['meta']['match_rate_fit']:.1%}/{r['meta']['match_rate_holdout']:.1%} "
              f"n_features={r['n_features']} "
              f"| baseline={base:.1f} trackman={r['holdout_score']:.1f} delta={delta:+.1f}")
        results.append((target_season, base, r["holdout_score"], delta))

    base_scores = np.array([b for _, b, _, _ in results])
    tm_scores = np.array([t for _, _, t, _ in results])
    print("\n=== SUMMARY ===")
    print(f"baseline: mean={base_scores.mean():.1f} std={base_scores.std():.1f}")
    print(f"trackman: mean={tm_scores.mean():.1f} std={tm_scores.std():.1f}")
    print(f"mean delta={tm_scores.mean()-base_scores.mean():+.1f}  "
          f"folds improved={ (tm_scores>base_scores).sum() }/5")


if __name__ == "__main__":
    main()
