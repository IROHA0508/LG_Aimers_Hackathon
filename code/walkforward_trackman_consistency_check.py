# walkforward_trackman_consistency_check.py
# ------------------------------------------------------------
# 트랙맨 이력을 "스터프 평균"(구속/회전/무브먼트/구종비율 평균) 대신
# "제구 일관성"(릴리스포인트/익스텐션/패스트볼 무브먼트-구속-회전의 표준편차)
# 피처로 재가공해서 다시 시도한다.
#
# 배경: 기존 시도(trackman_pitcher_lib.attach_trackman_pitcher_features, 팀/개인
# 레벨 둘 다)는 walk-forward에서 손해였다(README "세션 2" 참고). trackman에는
# 로케이션/성공여부가 없고 구질 물리량(평균)만 있어 asof_pitcher_success_rate류
# 이력보다 타겟과 간접적으로만 연결된다는 게 유력한 원인으로 추정됐다.
# "평균(무엇을 던지나)" 대신 "분산(얼마나 똑같이 던지나=커맨드 대리지표)"을 쓰면
# asof_* 성공률 이력과 덜 겹치는 정보가 될 수 있다는 가설을 이 스크립트로 검증한다.
#
# 비교 기준은 이번 세션에 새로 확정한 베이스라인(team/team_matchup/season_trend
# 피처 OFF, LGB 단독) walk-forward 5-fold 점수 [286.6, 1422.3, 2351.3, 0.0, 457.8]
# 평균 903.6 (walkforward_stacking_check.py의 LGB_ALONE_SCORES와 동일 출처).
# 기존 walkforward_trackman_check.py는 아직 팀/매치업/시즌추세 피처가 켜져 있던
# 구 베이스라인(706.0)과 비교했던 것이라 지금 기준으로는 재사용 불가.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from trackman_pitcher_lib import attach_trackman_consistency_features
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR

TARGET_COL = te.TARGET_COL

FEATURE_KWARGS = dict(use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False)
LGB_ALONE_SCORES = {2020: 286.6, 2021: 1422.3, 2022: 2351.3, 2023: 0.0, 2024: 457.8}


def run_one_fold(train, tm, target_season, n_internal_folds=5):
    fit_df = train[train["season"] < target_season].copy()
    holdout_df = train[train["season"] == target_season].copy()
    cutoff_season = target_season - 1

    fit_df, holdout_df, meta = attach_trackman_consistency_features(fit_df, holdout_df, tm, cutoff_season)

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
        r = run_one_fold(train, tm, target_season)
        base = LGB_ALONE_SCORES[target_season]
        delta = r["holdout_score"] - base
        print(f"predict {target_season}: n_mapped={r['meta']['n_mapped']} "
              f"match_rate(fit/holdout)={r['meta']['match_rate_fit']:.1%}/{r['meta']['match_rate_holdout']:.1%} "
              f"n_features={r['n_features']} "
              f"| baseline={base:.1f} trackman_consistency={r['holdout_score']:.1f} delta={delta:+.1f}")
        results.append((target_season, base, r["holdout_score"], delta))

    base_scores = np.array([b for _, b, _, _ in results])
    tm_scores = np.array([t for _, _, t, _ in results])
    print("\n=== SUMMARY ===")
    print(f"baseline (team/matchup/season OFF, LGB alone): mean={base_scores.mean():.1f}")
    print(f"+ trackman consistency features: mean={tm_scores.mean():.1f} std={tm_scores.std():.1f}")
    print(f"mean delta={tm_scores.mean()-base_scores.mean():+.1f}  "
          f"folds improved={ (tm_scores>base_scores).sum() }/5")


if __name__ == "__main__":
    main()
