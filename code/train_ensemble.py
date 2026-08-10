# train_ensemble.py
# ------------------------------------------------------------
# 투구 제구 성공 확률 예측 - 학습 파이프라인
# 전략: LightGBM + XGBoost + CatBoost 앙상블
#       -> OOF 예측 생성 -> Isotonic 확률보정 -> Logistic 메타러너 스태킹
# 평가지표(Brier Skill Score)는 판별력보다 "보정 품질"에 민감하므로
# 보정 단계를 파이프라인 필수 스텝으로 둔다.
#
# 주의: 이 스크립트는 참가자 로컬/학습 환경에서 실행하는 "학습용" 코드다.
#       대회의 10분 추론 제한은 script.py(추론 전용)에만 적용되며,
#       여기서 만든 모델 가중치를 model/ 디렉토리에 저장해 제출한다.
# ------------------------------------------------------------
import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

ID_COL = "row_id"
TARGET_COL = "control_success"
N_FOLDS = 5
SEED = 42

CAT_COLS = ["top_bottom", "game_type", "base_state",
            "pitcher_hand", "batter_hand"]

RATE_COLS_FOR_MISSING_FLAG = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


# ============================================================
# 1) 트랙맨 이력 -> 투수 단위 "career prior" 룩업 테이블 생성
#    - 시점 리크 방지를 위해 반드시 train 데이터의 최소 시즌보다
#      이전 구간만 사용하거나, 최소한 game_date 기준 컷오프를 둔다.
#    - 평가 서버는 data/ 에 trackman_history.csv를 다시 안 줄 수도
#      있으므로, 학습 시점에 미리 룩업 테이블로 "고정"해서
#      model 번들에 함께 저장한다 (script.py는 트랙맨 원본 파일 불필요).
# ============================================================

def build_trackman_pitcher_prior(trackman_path, cutoff_season=None):
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig")

    if cutoff_season is not None:
        tm = tm[tm["season"] <= cutoff_season]

    # 그룹별(구종군) 사용비율 -> 투수 단위 wide 피처
    grp = tm.groupby("pitcher_trackman_id")
    prior = grp.agg(
        tm_n=("pitch_type_group", "size"),
        tm_rel_speed_mean=("rel_speed", "mean"),
        tm_spin_rate_mean=("spin_rate", "mean"),
        tm_ivb_mean=("induced_vert_break", "mean"),
        tm_hb_mean=("horz_break", "mean"),
        tm_extension_mean=("extension", "mean"),
    ).reset_index()

    mix = (tm.groupby(["pitcher_trackman_id", "pitch_type_group"])
             .size().unstack(fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0).add_prefix("tm_mix_")
    prior = prior.merge(mix.reset_index(), on="pitcher_trackman_id", how="left")
    prior = prior.rename(columns={"pitcher_trackman_id": "pitcher_id"})
    return prior


def attach_trackman_prior(df, prior_table):
    """pitcher_id 조인. ID 네임스페이스가 다르면 자동으로 매칭률이 낮게
    나오므로, 학습 스크립트 실행 후 매칭률을 반드시 로그로 확인할 것."""
    before_cols = set(df.columns)
    df = df.merge(prior_table, on="pitcher_id", how="left")
    new_cols = [c for c in df.columns if c not in before_cols]
    match_rate = df[new_cols[0]].notna().mean() if new_cols else 0.0
    print(f"  [trackman prior] 매칭률: {match_rate:.1%} "
          f"(낮으면 pitcher_id <-> pitcher_trackman_id 매핑을 별도 확인 필요)")
    return df, new_cols


# ============================================================
# 2) 피처 엔지니어링
# ============================================================

def build_features(df, trackman_prior=None):
    df = df.copy()

    # cold-start(표본 0) 결측 플래그 - "정보 없음" 자체가 신호일 수 있음
    for c in RATE_COLS_FOR_MISSING_FLAG:
        if c in df.columns:
            df[f"{c}_isna"] = df[c].isna().astype(np.int8)

    # 표본 수 기반 신뢰도 가중 (n이 작을수록 rate의 노이즈가 큼)
    if "asof_pitcher_n" in df.columns:
        df["asof_pitcher_n_log"] = np.log1p(df["asof_pitcher_n"])
    if "asof_batter_n" in df.columns:
        df["asof_batter_n_log"] = np.log1p(df["asof_batter_n"])
    if "asof_pitcher_pitchmix_n" in df.columns:
        df["asof_pitcher_pitchmix_n_log"] = np.log1p(df["asof_pitcher_pitchmix_n"])

    # 카운트 파생 (구종 예측/압박 상황과 상관)
    if {"balls_before", "strikes_before"}.issubset(df.columns):
        df["count_pressure"] = df["strikes_before"] - df["balls_before"]
        df["is_two_strike"] = (df["strikes_before"] == 2).astype(np.int8)
        df["is_three_ball"] = (df["balls_before"] == 3).astype(np.int8)

    # 트랙맨 커리어 prior 조인 (선택)
    if trackman_prior is not None:
        df, _ = attach_trackman_prior(df, trackman_prior)

    # 범주형 -> category dtype (LightGBM/CatBoost native 지원)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    drop_cols = [ID_COL, TARGET_COL]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return df, feat_cols


# ============================================================
# 3) 모델별 학습 함수 (OOF 반환)
# ============================================================

def train_lgb(X, y, X_full, cat_features, n_folds=N_FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=5.0, max_depth=-1, verbose=-1, seed=seed,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tr_set = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_features)
        va_set = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_features)
        model = lgb.train(params, tr_set, num_boost_round=4000,
                           valid_sets=[va_set],
                           callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        models.append(model)
        print(f"  [LGB fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


def train_xgb(X, y, cat_features, n_folds=N_FOLDS, seed=SEED):
    # XGBoost native categorical 지원 (enable_categorical=True)
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = dict(
        objective="binary:logistic", eval_metric="logloss",
        eta=0.03, max_depth=8, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        tree_method="hist", enable_categorical=True, seed=seed,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        model = xgb.train(params, dtr, num_boost_round=4000,
                           evals=[(dva, "valid")],
                           early_stopping_rounds=200, verbose_eval=False)
        oof[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        models.append(model)
        print(f"  [XGB fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


def train_cat(X, y, cat_features, n_folds=N_FOLDS, seed=SEED):
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tr_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
        va_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_features)
        model = CatBoostClassifier(
            iterations=4000, learning_rate=0.03, depth=8,
            l2_leaf_reg=5.0, loss_function="Logloss",
            eval_metric="Logloss", random_seed=seed,
            early_stopping_rounds=200, verbose=False,
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(model)
        print(f"  [CAT fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


# ============================================================
# 4) 메인
# ============================================================

def main():
    DATA_DIR = "./data"
    MODEL_DIR = "./model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Load train...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    y = train[TARGET_COL].values

    print("Build trackman prior (있으면 사용, 없으면 스킵)...")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    prior_table = None
    if os.path.exists(trackman_path):
        # 학습 데이터에 여러 시즌이 섞여 있다면, 시즌별로 컷오프를 두고
        # prior를 다시 계산하는 것이 정석이지만(진짜 leak-free),
        # 여기서는 단순화를 위해 전체 이력을 하나의 "커리어 prior"로 사용한다.
        # -> 실제 대회에서는 season 컬럼 기준으로 asof 방식으로 정교화 권장.
        prior_table = build_trackman_pitcher_prior(trackman_path)

    print("Build features...")
    train_feat, feat_cols = build_features(train, prior_table)
    cat_features = [c for c in CAT_COLS if c in feat_cols]

    X = train_feat[feat_cols]
    print(f"  n_features={len(feat_cols)}, n_rows={len(X)}")

    print("Train LightGBM...")
    lgb_models, lgb_oof = train_lgb(X, y, X, cat_features)

    print("Train XGBoost...")
    xgb_models, xgb_oof = train_xgb(X, y, cat_features)

    print("Train CatBoost...")
    cat_models, cat_oof = train_cat(X, y, cat_features)

    print("Calibrate each model's OOF (Isotonic)...")
    calibrators = {}
    calibrated = {}
    for name, oof in [("lgb", lgb_oof), ("xgb", xgb_oof), ("cat", cat_oof)]:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y)
        calibrators[name] = iso
        calibrated[name] = iso.predict(oof)
        print(f"  [{name}] raw brier={brier_score_loss(y, oof):.5f} "
              f"-> calibrated brier={brier_score_loss(y, calibrated[name]):.5f}")

    print("Stack (meta Logistic Regression on calibrated OOF probs)...")
    meta_X = np.column_stack([calibrated["lgb"], calibrated["xgb"], calibrated["cat"]])
    meta = LogisticRegression()
    meta.fit(meta_X, y)
    stack_pred = meta.predict_proba(meta_X)[:, 1]
    print(f"  stacked brier(보정 전)={brier_score_loss(y, stack_pred):.5f}")

    # 스태킹 결과 자체도 한번 더 보정 (Brier Skill Score는 보정에 매우 민감)
    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(stack_pred, y)
    final_pred = final_iso.predict(stack_pred)
    print(f"  stacked brier(최종 보정 후)={brier_score_loss(y, final_pred):.5f}")

    print("Save model bundle...")
    bundle = {
        "lgb_models": lgb_models,
        "xgb_models": xgb_models,
        "cat_models": cat_models,
        "calibrators": calibrators,
        "meta_model": meta,
        "final_calibrator": final_iso,
        "feat_cols": feat_cols,
        "cat_features": cat_features,
        "trackman_prior_table": prior_table,  # inference 시 원본 트랙맨 파일 불필요
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "ensemble_bundle.pkl"))
    print("✅ Saved model/ensemble_bundle.pkl")


if __name__ == "__main__":
    main()
