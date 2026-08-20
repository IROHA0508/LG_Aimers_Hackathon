# walkforward_trackman_consistency_stack_check.py
# ------------------------------------------------------------
# walkforward_trackman_consistency_check.py에서 LGB 단독 기준으로 트랙맨
# "일관성"(표준편차) 피처가 5-fold walk-forward 평균 +9.6(903.6->913.2,
# 3/5 폴드 개선)로 약하지만 처음으로 순긍정 신호를 보였다. 기존 트랙맨
# 시도(스터프 평균)는 walk-forward에서 전부 손해였던 것과 대비된다.
#
# 다만 +9.6은 폴드별 변동폭(수백~수천)에 비해 작고, 실제 배포 후보는 LGB
# 단독이 아니라 LGB+RF+MLP 3way 스태킹(934.0)이다. 스태킹 단계에서까지
# 이 효과가 유지되는지, 아니면 다른 모델의 오차와 섞이며 사라지는지
# 확인해야 최종 채택 여부를 판단할 수 있다.
#
# 같은 스크립트 안에서 트랙맨 피처 유무 두 버전을 모두 처음부터 학습해
# (레퍼런스 점수 재사용에 따른 실행 시점 불일치 위험 없이) 폴드별로 직접 비교한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from trackman_pitcher_lib import build_pitcher_id_map, build_trackman_consistency_prior
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR

TARGET_COL = te.TARGET_COL
FEATURE_KWARGS = dict(use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False)


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


def run_fold_3way(train, target_season, tm=None):
    X_fit, y_fit, X_holdout, y_holdout, cat_features = prep_fold(train, target_season, tm=tm)

    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    lgb_pred_h = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)

    mlp_state_dicts, mlp_oof, mlp_preproc = te.train_mlp(X_fit, y_fit, cat_features, verbose_fold=False)
    mlp_pred_h = te.predict_mlp(mlp_state_dicts, mlp_preproc, X_holdout)

    rf_models, rf_oof = te.train_rf(X_fit, y_fit, cat_features)
    rf_pred_h = np.mean([pipe.predict_proba(X_holdout)[:, 1] for pipe in rf_models], axis=0)

    pred = fit_stack({"lgb": lgb_oof, "rf": rf_oof, "mlp": mlp_oof}, y_fit,
                      {"lgb": lgb_pred_h, "rf": rf_pred_h, "mlp": mlp_pred_h})
    return score(brier_score_loss(y_holdout, pred))


def main():
    print("Load data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    tm["team_merged"] = tm["pitcher_team"].replace({"SSG_LAN": "SK_WYV"})

    results = []
    for target_season in FOLD_SEASONS:
        base = run_fold_3way(train, target_season, tm=None)
        with_tm = run_fold_3way(train, target_season, tm=tm)
        delta = with_tm - base
        print(f"predict {target_season}: LGB+RF+MLP base={base:.1f}  +trackman_consistency={with_tm:.1f}  delta={delta:+.1f}")
        results.append((target_season, base, with_tm, delta))

    base_scores = np.array([b for _, b, _, _ in results])
    tm_scores = np.array([t for _, _, t, _ in results])
    print("\n=== SUMMARY ===")
    print(f"LGB+RF+MLP base:              mean={base_scores.mean():.1f}")
    print(f"LGB+RF+MLP + trackman consist: mean={tm_scores.mean():.1f}")
    print(f"mean delta={tm_scores.mean()-base_scores.mean():+.1f}  folds improved={(tm_scores>base_scores).sum()}/5")


if __name__ == "__main__":
    main()
