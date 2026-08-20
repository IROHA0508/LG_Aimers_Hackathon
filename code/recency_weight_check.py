# recency_weight_check.py
# ------------------------------------------------------------
# 남은 후보 중 하나: 오래된 시즌 데이터를 통째로 버리는 대신(레짐분리, 이미
# 기각됨), 최근 시즌에 sample weight를 지수적으로 더 주는 recency-weighted
# 학습. train_ensemble.train_lgb는 weight 인자가 없어서 이 스크립트에서
# 같은 구조를 (weight 지원 추가해) 별도로 구현한다. 튜닝된 하이퍼파라미터/
# early stopping 설정은 그대로 재사용.
#
# weight = decay_rate ** (max_fit_season - season), 즉 fit 마지막 시즌(2023)의
# 가중치는 항상 1.0, 그 이전 시즌일수록 decay_rate배씩 감소.
# cross-regime harness(2019-2023 학습 -> 2024 홀드아웃) 그대로 사용.
# baseline(균등 가중치) holdout score = 460.2 (time_holdout_check.py 결과)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
BASELINE_BRIER = 0.249446

LGB_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.005, num_leaves=320, min_data_in_leaf=800,
    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, max_depth=-1, verbose=-1, seed=te.SEED,
)

DECAY_RATES = [0.6, 0.3]


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def train_lgb_weighted(X, y, w, cat_features, n_folds=te.N_FOLDS, seed=te.SEED,
                        num_boost_round=8000, early_stopping_rounds=300):
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tr_set = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], weight=w[tr_idx],
                              categorical_feature=cat_features)
        va_set = lgb.Dataset(X.iloc[va_idx], y[va_idx], weight=w[va_idx],
                              categorical_feature=cat_features)
        model = lgb.train(LGB_PARAMS, tr_set, num_boost_round=num_boost_round,
                           valid_sets=[va_set],
                           callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        models.append(model)
    return models, oof


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()
    print(f"fit={len(fit_df)}  holdout={len(holdout_df)}")

    snapshot_tables = te.build_train_snapshot_tables(fit_df)
    fit_feat, feat_cols = te.build_features(fit_df, snapshot_tables=None)
    cat_features = [c for c in te.CAT_COLS if c in feat_cols]
    cat_dtype_categories = {c: fit_feat[c].cat.categories for c in cat_features}

    y_holdout = holdout_df[TARGET_COL].values
    holdout_input = holdout_df.drop(columns=[TARGET_COL])
    holdout_feat, _ = te.build_features(
        holdout_input, snapshot_tables=snapshot_tables,
        cat_dtype_categories=cat_dtype_categories)

    X_fit = fit_feat[feat_cols]
    y_fit = fit_df[TARGET_COL].values
    X_holdout = holdout_feat[feat_cols]
    fit_season = fit_df["season"].values
    max_fit_season = fit_season.max()

    results = []
    for decay in DECAY_RATES:
        w = decay ** (max_fit_season - fit_season).astype(float)
        print(f"\n=== decay_rate={decay} (weight range: {w.min():.4f} ~ {w.max():.4f}) ===")
        print("  weight by season:", {int(s): round(decay ** (max_fit_season - s), 4)
                                        for s in sorted(set(fit_season))})

        models, oof = train_lgb_weighted(X_fit, y_fit, w, cat_features)
        fit_oof_brier = brier_score_loss(y_fit, oof)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)
        fit_cal_brier = brier_score_loss(y_fit, iso.predict(oof))

        holdout_pred = np.mean(
            [m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
        holdout_brier = brier_score_loss(y_holdout, holdout_pred)
        holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred))

        print(f"  fit OOF(calibrated) score={score(fit_cal_brier):.1f}")
        print(f"  holdout(raw) score={score(holdout_brier):.1f}  holdout(calibrated) score={score(holdout_cal_brier):.1f}")
        results.append((decay, score(fit_cal_brier), score(holdout_brier), score(holdout_cal_brier)))

    print("\n\n=== SUMMARY (baseline uniform-weight holdout score = 460.2) ===")
    print(f"{'decay_rate':12s} {'fit_score':>12s} {'holdout(raw)':>14s} {'holdout(cal)':>14s}")
    for decay, a, b, c in results:
        print(f"{decay:<12} {a:12.1f} {b:14.1f} {c:14.1f}")


if __name__ == "__main__":
    main()
