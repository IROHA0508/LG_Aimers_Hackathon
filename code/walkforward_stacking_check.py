# walkforward_stacking_check.py
# ------------------------------------------------------------
# 세션1이 랜덤 K-fold OOF 상관관계 논리로 "LGB+MLP 2way 스태킹"을 최종
# 채택했는데(LGB 단독 대비 +13.7점), 이게 forward-time(연도별 walk-forward)
# 기준으로도 유효한지 재검증한다.
#
# 피처 설정은 이번 세션에서 확정한 개선안(team/team_matchup/season_trend
# 피처 OFF)을 기준으로 사용 - 그래야 실제 배포할 파이프라인과 같은 조건에서
# 스태킹 이득을 판단할 수 있다. LGB 단독(같은 피처 설정) 5-fold walk-forward
# 결과는 이미 확보함: [286.6, 1422.3, 2351.3, 0.0, 457.8] 평균 903.6
# (walkforward_feature_ablation2.py 결과).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR

TARGET_COL = te.TARGET_COL

FEATURE_KWARGS = dict(use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False)
LGB_ALONE_SCORES = {2020: 286.6, 2021: 1422.3, 2022: 2351.3, 2023: 0.0, 2024: 457.8}


def prep_fold(train, target_season):
    fit_df = train[train["season"] < target_season].copy()
    holdout_df = train[train["season"] == target_season].copy()

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


def run_stack_fold(train, target_season):
    X_fit, y_fit, X_holdout, y_holdout, cat_features = prep_fold(train, target_season)

    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    mlp_state_dicts, mlp_oof, mlp_preproc = te.train_mlp(X_fit, y_fit, cat_features, verbose_fold=False)

    calibrators = {}
    calibrated = {}
    for name, oof in [("lgb", lgb_oof), ("mlp", mlp_oof)]:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)
        calibrators[name] = iso
        calibrated[name] = iso.predict(oof)

    meta_X = np.column_stack([calibrated["lgb"], calibrated["mlp"]])
    meta = LogisticRegression()
    meta.fit(meta_X, y_fit)
    stack_pred_fit = meta.predict_proba(meta_X)[:, 1]

    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(stack_pred_fit, y_fit)

    lgb_pred_h = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)
    mlp_pred_h = te.predict_mlp(mlp_state_dicts, mlp_preproc, X_holdout)
    lgb_pred_h_cal = calibrators["lgb"].predict(lgb_pred_h)
    mlp_pred_h_cal = calibrators["mlp"].predict(mlp_pred_h)

    meta_X_h = np.column_stack([lgb_pred_h_cal, mlp_pred_h_cal])
    stack_pred_h = meta.predict_proba(meta_X_h)[:, 1]
    final_pred_h = final_iso.predict(stack_pred_h)

    lgb_only_brier = brier_score_loss(y_holdout, lgb_pred_h_cal)
    mlp_only_brier = brier_score_loss(y_holdout, mlp_pred_h_cal)
    stack_brier = brier_score_loss(y_holdout, final_pred_h)

    corr = np.corrcoef(lgb_oof, mlp_oof)[0, 1]
    return dict(target_season=target_season,
                lgb_only=score(lgb_only_brier), mlp_only=score(mlp_only_brier),
                stack=score(stack_brier), oof_corr=corr)


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    results = []
    for target_season in FOLD_SEASONS:
        r = run_stack_fold(train, target_season)
        lgb_ref = LGB_ALONE_SCORES[target_season]
        print(f"predict {target_season}: oof_corr(lgb,mlp)={r['oof_corr']:.3f} | "
              f"LGB-alone(ref)={lgb_ref:.1f}  LGB-here={r['lgb_only']:.1f}  MLP-alone={r['mlp_only']:.1f}  "
              f"STACK={r['stack']:.1f}  stack_vs_lgb_delta={r['stack']-lgb_ref:+.1f}")
        results.append(r)

    lgb_scores = np.array([LGB_ALONE_SCORES[r["target_season"]] for r in results])
    mlp_scores = np.array([r["mlp_only"] for r in results])
    stack_scores = np.array([r["stack"] for r in results])
    print("\n=== SUMMARY ===")
    print(f"LGB alone:  mean={lgb_scores.mean():.1f}")
    print(f"MLP alone:  mean={mlp_scores.mean():.1f}")
    print(f"LGB+MLP stack: mean={stack_scores.mean():.1f}  delta_vs_lgb={stack_scores.mean()-lgb_scores.mean():+.1f}")
    print(f"folds where stack beat LGB alone: {(stack_scores>lgb_scores).sum()}/5")


if __name__ == "__main__":
    main()
