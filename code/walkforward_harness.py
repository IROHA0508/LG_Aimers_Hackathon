# walkforward_harness.py
# ------------------------------------------------------------
# 지금까지 쓴 단일 분할(2019-2023 학습 -> 2024 검증)은 표본이 1개뿐이라
# 결과가 노이즈인지 실제 효과인지 판단하기 애매한 경우가 많았다(정규화
# 실험이 대표적 사례: 작은 표본/큰 표본에서 결론이 뒤집힘).
#
# 이 스크립트는 expanding-window walk-forward 검증을 구현한다:
#   fold 1: 2019      학습 -> 2020 예측
#   fold 2: 2019-2020 학습 -> 2021 예측
#   fold 3: 2019-2021 학습 -> 2022 예측
#   fold 4: 2019-2022 학습 -> 2023 예측
#   fold 5: 2019-2023 학습 -> 2024 예측
# 5개 폴드의 holdout score를 평균+표준편차로 보고해서 노이즈 여부를
# 판단할 수 있게 한다. 피처 on/off 실험(build_features의 use_* 플래그)도
# 이 harness로 바로 테스트 가능하도록 파라미터화했다.
#
# 매 폴드 features/target 조합이 바뀌므로 build_features를 매번 새로
# 호출해야 한다(캐싱 안 함 - 폴드 수가 적어 재계산 비용이 크지 않음).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
BASELINE_BRIER = 0.249446

FOLD_SEASONS = [2020, 2021, 2022, 2023, 2024]  # 각각 예측 대상 시즌(fit=그 이전 전부)


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def run_one_fold(train, target_season, feature_kwargs, n_internal_folds):
    fit_df = train[train["season"] < target_season].copy()
    holdout_df = train[train["season"] == target_season].copy()

    snapshot_tables = te.build_train_snapshot_tables(fit_df)
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None, **feature_kwargs)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories, **feature_kwargs)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]

    models, oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features,
                                n_folds=n_internal_folds, verbose_fold=False)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_fit)

    holdout_pred = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
    holdout_brier = brier_score_loss(y_holdout, holdout_pred)
    holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred))

    return dict(target_season=target_season, n_fit=len(X_fit), n_holdout=len(X_holdout),
                n_features=len(feat_cols),
                holdout_score_raw=score(holdout_brier), holdout_score_cal=score(holdout_cal_brier))


def walk_forward_eval(train, feature_kwargs=None, n_internal_folds=5, fold_seasons=None, name="config"):
    feature_kwargs = feature_kwargs or {}
    fold_seasons = fold_seasons or FOLD_SEASONS
    print(f"\n########## walk-forward eval: {name}  feature_kwargs={feature_kwargs} "
          f"n_internal_folds={n_internal_folds} ##########")
    rows = []
    for target_season in fold_seasons:
        r = run_one_fold(train, target_season, feature_kwargs, n_internal_folds)
        print(f"  [{name}] predict {target_season} (fit n={r['n_fit']}, features={r['n_features']}): "
              f"holdout score(raw)={r['holdout_score_raw']:.1f}  score(calibrated)={r['holdout_score_cal']:.1f}")
        rows.append(r)

    cal_scores = np.array([r["holdout_score_cal"] for r in rows])
    print(f"  [{name}] SUMMARY: mean={cal_scores.mean():.1f}  std={cal_scores.std():.1f}  "
          f"min={cal_scores.min():.1f}  max={cal_scores.max():.1f}")
    return rows


if __name__ == "__main__":
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    walk_forward_eval(train, feature_kwargs={}, n_internal_folds=5, name="baseline (current default features)")
