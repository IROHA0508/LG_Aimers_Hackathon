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

# 트랙맨 이력 사용 여부.
# pitcher_id(train.csv) <-> pitcher_trackman_id(trackman_history.csv) 매칭률을
# 실제로 확인한 결과 0.0%로, 두 ID가 서로 다른 익명화 공간이라 연결이 불가능함이
# 확인됐다 (탐색 노트북 참고). 따라서 기본값을 False로 둔다.
# 추후 대회 측에서 매핑 정보를 제공하는 등 상황이 바뀌면 True로 전환.
USE_TRACKMAN = False

# GPU 사용 여부. 로컬(RTX 4050)에서 XGBoost/CatBoost GPU 동작을 확인했으므로 기본 True.
# LightGBM은 기본 pip 설치본에 GPU가 안 켜져 있어(별도 빌드 필요) 기본 False 권장.
# 아래 각 train_* 함수는 GPU 시도 실패 시 자동으로 CPU로 재시도하도록 안전장치를 넣었다.
USE_GPU_XGB = True
USE_GPU_CAT = True
USE_GPU_LGB = False

CAT_COLS = ["top_bottom", "game_type", "base_state",
            "pitcher_hand", "batter_hand", "hand_matchup"]

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

def build_matchup_features(df, shrink_k=10):
    """투수-타자 맞대결 as-of 피처 (leak-free).

    사전 검증 결과:
    - row_id 순서가 실제 시간순임을 확인함 (pitcher_id별 asof_pitcher_n이
      row 순서대로 단조증가, 역행 0건).
    - 전체 행의 93.5%가 이미 해당 (투수,타자) 조합의 재대결이고
      (재대결 행 기준 중앙값 11회, 평균 20.4회), 1회성 조합은 3.9%뿐이라
      콜드스타트 비율이 낮아 신호가 충분히 있을 것으로 기대됨.

    누수 방지: (pitcher_id, batter_id) 그룹 내에서 "현재 행 이전"의
    누적 횟수/성공수만 사용한다(cumcount는 현재 행을 세지 않고, cumsum은
    현재 행의 target을 빼서 제외). 표본이 적은 조합은 노이즈가 크므로
    해당 투수의 전체 asof_pitcher_success_rate로 축소추정(shrinkage)한다.

    주의(TODO): 이 함수는 라벨(control_success)이 있고 시간순 정렬된
    데이터에만 적용 가능하다. 실제 test.csv에는 라벨이 없고, 대회 규정상
    test.csv 행끼리 조합해 피처를 만드는 것도 금지이므로, 제출 파이프라인
    (script.py)에서는 이 함수 대신 train.csv 이력만으로 만든
    (pitcher_id, batter_id) 정적 lookup 테이블을 test.csv에 조인하는
    방식을 별도로 구현해야 한다. 지금은 탐색 노트북에서 이 피처가
    실제로 도움이 되는지 먼저 검증하는 단계다.
    """
    if TARGET_COL not in df.columns:
        return df

    df = df.sort_values(ID_COL).reset_index(drop=True)

    grp_key = ["pitcher_id", "batter_id"]
    matchup_n = df.groupby(grp_key).cumcount()
    cum_success = df.groupby(grp_key)[TARGET_COL].cumsum() - df[TARGET_COL]

    pitcher_prior = df["asof_pitcher_success_rate"].fillna(df[TARGET_COL].mean())

    df["matchup_n"] = matchup_n
    df["matchup_n_log"] = np.log1p(matchup_n)
    df["matchup_success_rate"] = cum_success / matchup_n.replace(0, np.nan)
    df["matchup_success_rate_isna"] = (matchup_n == 0).astype(np.int8)
    df["matchup_success_rate_shrunk"] = (
        (cum_success + shrink_k * pitcher_prior) / (matchup_n + shrink_k)
    )
    return df


def build_team_features(df):
    """팀 단위 as-of 피처 (leak-free).

    선수 개인 단위(asof_pitcher_*, asof_batter_*)와 별개로 팀 전체 단위의
    과거 경향(코칭 스타일, 구장 요인, 포수진 등 개인 통계로는 못 잡는
    조직 단위 신호)을 노려본다. 팀 ID는 13개뿐이라 표본이 매우 커서
    (팀당 평균 11만행) 콜드스타트 영향은 거의 없다고 봐도 된다.

    pitcher_team_id는 시즌 중 이적으로 바뀔 수 있음을 확인했다(선수의 20.7%가
    커리어 중 팀ID 2개 이상). team_id 기준으로 그룹을 나누므로 이적 시점에
    자동으로 그 팀의 누적치로 전환되어 별도 처리가 필요 없다.

    matchup 피처와 동일하게 (그룹 내 "현재 행 이전"만 사용) 누수를 방지한다.
    """
    if TARGET_COL not in df.columns:
        return df

    df = df.sort_values(ID_COL).reset_index(drop=True)

    for team_col, prefix in [("pitcher_team_id", "pitcher_team"),
                              ("batter_team_id", "batter_team")]:
        if team_col not in df.columns:
            continue
        n = df.groupby(team_col).cumcount()
        cum_success = df.groupby(team_col)[TARGET_COL].cumsum() - df[TARGET_COL]
        df[f"asof_{prefix}_n"] = n
        df[f"asof_{prefix}_success_rate"] = cum_success / n.replace(0, np.nan)

    return df


def build_team_matchup_features(df):
    """팀x팀(투수팀 vs 타자팀) as-of 맞대결 피처.

    투수-타자 개별 맞대결(build_matchup_features)은 조합이 96,133개라
    조합당 평균 15행뿐이라 실패했지만, 팀 단위로 묶으면 조합이 96개뿐이라
    조합당 평균 15,365행(최소 292행)으로 표본 규모가 완전히 다르다.
    team 단위 피처(build_team_features)가 성공했던 것과 같은 이유로
    성공 가능성을 기대해볼 수 있다.
    """
    if TARGET_COL not in df.columns:
        return df

    df = df.sort_values(ID_COL).reset_index(drop=True)

    grp_key = ["pitcher_team_id", "batter_team_id"]
    n = df.groupby(grp_key).cumcount()
    cum_success = df.groupby(grp_key)[TARGET_COL].cumsum() - df[TARGET_COL]
    df["asof_team_matchup_n"] = n
    df["asof_team_matchup_success_rate"] = cum_success / n.replace(0, np.nan)

    return df


def build_season_trend_features(df):
    """시즌 기준선 대비 상대 성공률 (era-adjusted rate).

    실측 결과 시즌별 control_success 비율이 2019 0.565 -> 2024 0.486로
    6년간 7.9%p 꾸준히 하락하는 큰 추세가 있다. asof_pitcher/batter_success_rate는
    커리어 누적 평균이라 이 추세가 섞여 있어(예: 2019~2024를 다 뛴 베테랑은
    성공률 높았던 초기 시즌과 낮아진 최근 시즌이 뭉개진 평균), "이 선수가 리그
    평균 대비 실제로 잘하는지"를 흐릴 수 있다. 시즌 자체의 as-of 기준선을
    별도로 만들어 빼주면 선수 개인의 순수 편차를 더 선명하게 분리할 수 있다.

    시즌 그룹 크기가 시즌당 평균 24만행으로 매우 커서 콜드스타트(시즌 첫 몇 행)는
    무시할 수준이다.
    """
    if TARGET_COL not in df.columns:
        return df

    df = df.sort_values(ID_COL).reset_index(drop=True)

    n = df.groupby("season").cumcount()
    cum_success = df.groupby("season")[TARGET_COL].cumsum() - df[TARGET_COL]
    season_rate = cum_success / n.replace(0, np.nan)

    df["asof_season_success_rate"] = season_rate
    if "asof_pitcher_success_rate" in df.columns:
        df["pitcher_rate_vs_season"] = df["asof_pitcher_success_rate"] - season_rate
    if "asof_batter_success_rate" in df.columns:
        df["batter_rate_vs_season"] = df["asof_batter_success_rate"] - season_rate

    return df


def build_clutch_features(df, shrink_k=30):
    """투수별 압박 상황(풀카운트) 클러치 성향 as-of 피처.

    "상황 자체가 압박인가"는 is_two_strike/is_three_ball/count_pressure로 이미
    다 잡혀있어서(is_full_count를 별도로 줘봐도 순수 논리적 중복이라 기각됨),
    "이 투수가 그 상황에서 유독 강한지/약한지"는 다른 피처가 커버 못하는
    선수 정체성 x 상황의 교차 정보다.

    사전 검증: 압박(풀카운트) 겪은 투수 762명, 투수당 표본 중앙값 47행(25%는
    10행 미만)으로 얇은 편이라 노이즈 위험이 있다. asof_pitcher_success_rate로
    강하게 축소추정(k=30, 맞대결 피처의 k=10보다 강함)해서 저표본 투수는
    자기 평균 쪽으로 세게 당긴다.

    풀카운트가 아닌 행에는 정의상 의미가 없으므로 NaN으로 둔다(LGB가 네이티브 처리).
    """
    if TARGET_COL not in df.columns:
        return df

    df = df.sort_values(ID_COL).reset_index(drop=True)

    is_pressure = (df["balls_before"] == 3) & (df["strikes_before"] == 2)
    df["_pressure_tmp"] = is_pressure.astype(np.int8)

    grp_key = ["pitcher_id", "_pressure_tmp"]
    n = df.groupby(grp_key).cumcount()
    cum_success = df.groupby(grp_key)[TARGET_COL].cumsum() - df[TARGET_COL]

    pitcher_prior = df["asof_pitcher_success_rate"].fillna(df[TARGET_COL].mean())
    pressure_rate_shrunk = (cum_success + shrink_k * pitcher_prior) / (n + shrink_k)
    clutch_delta = pressure_rate_shrunk - pitcher_prior

    df["pitcher_clutch_delta"] = np.where(is_pressure, clutch_delta, np.nan)
    df["pitcher_clutch_n"] = np.where(is_pressure, n, np.nan)

    df = df.drop(columns=["_pressure_tmp"])
    return df


def build_features(df, trackman_prior=None,
                    use_matchup_feature=False, matchup_shrink_k=10,
                    use_team_feature=True,
                    use_clutch_feature=False,
                    use_team_matchup_feature=True,
                    use_season_trend_feature=True):
    # use_matchup_feature: 투수-타자 맞대결 as-of 피처(build_matchup_features) 사용 여부.
    # feature_matchup.ipynb에서 검증한 결과 shrink_k(10, 100 모두)와 무관하게
    # 전체 데이터(147만행) 기준 LightGBM Brier가 오히려 소폭 악화됨(-0.07%)이 확인되어
    # 기본값을 False로 둔다. asof_pitcher/batter_success_rate + pitcher_id/batter_id의
    # 암묵적 조합으로 이미 커버되는 정보로 판단, 채택하지 않음.
    #
    # use_team_feature: 팀 단위 as-of 피처(build_team_features) 사용 여부.
    # feature_team.ipynb에서 검증한 결과 전체 데이터 기준 LightGBM Brier가
    # 0.24405 -> 0.24382로 개선(+0.0956%, 지금까지 시도한 피처/모델 변경 중 최대폭)되어
    # 기본값을 True로 둔다. Feature importance도 상위권(3위, 7위 등)으로 실제 기여 확인됨.
    #
    # use_team_matchup_feature / use_season_trend_feature: feature_teammatchup_season.ipynb에서
    # 검증한 결과(둘을 함께 넣고 확인) 전체 데이터 기준 LightGBM Brier가
    # 0.243817 -> 0.24376으로 개선(+0.0237%)되어 기본값을 True로 둔다.
    # Feature importance는 3위/4위로 높게 나오지만(season_success_rate, team_matchup_success_rate)
    # 실제 개선폭은 팀 피처보다 작음 -> 기존 season/team 피처와 정보가 일부 겹치는 것으로 추정.
    #
    # use_clutch_feature: feature_clutch.ipynb에서 검증한 결과 전체 데이터 기준
    # LightGBM Brier가 0.243760 -> 0.243790으로 악화됨(-0.0124%)이 확인되어
    # 기본값을 False로 둔다. 투수당 압박(풀카운트) 표본이 중앙값 47행(25%는 10행 미만)으로
    # 얇고, 상관계수도 0.0163으로 지금까지 시도한 피처 중 가장 약해서 shrink_k=30으로도
    # 노이즈를 못 이겼음. 맞대결 피처와 같은 "선수 정체성 x 희소 상황" 실패 패턴.
    df = df.copy()
    if use_matchup_feature:
        df = build_matchup_features(df, shrink_k=matchup_shrink_k)
    if use_team_feature:
        df = build_team_features(df)
    if use_clutch_feature:
        df = build_clutch_features(df)
    if use_team_matchup_feature:
        df = build_team_matchup_features(df)
    if use_season_trend_feature:
        df = build_season_trend_features(df)

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

    # 좌우 상성(platoon split) - pitcher_hand/batter_hand는 CAT_COLS에 개별로
    # 이미 들어있어 트리가 두 컬럼을 조합해서 암묵적으로 상성을 찾을 수도 있지만,
    # 4개 조합 조건부 성공률을 실측한 결과 최대 4.7%p 차이가 나서(0.4909~0.5375)
    # 명시적 조합 피처로 줘서 트리가 한 번의 분기로 바로 찾게 한다.
    if {"pitcher_hand", "batter_hand"}.issubset(df.columns):
        df["is_same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int8)
        df["hand_matchup"] = (
            df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
        )

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

def train_lgb(X, y, X_full, cat_features, n_folds=N_FOLDS, seed=SEED,
              params_override=None, early_stopping_rounds=200, verbose_fold=True,
              num_boost_round=4000):
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=5.0, max_depth=-1, verbose=-1, seed=seed,
    )
    if params_override:
        params.update(params_override)
    gpu_params = dict(params)
    if USE_GPU_LGB:
        # 이 옵션이 동작하려면 GPU 지원이 활성화된 LightGBM 빌드가 필요함
        # (기본 `pip install lightgbm`은 CPU 전용인 경우가 많음)
        gpu_params.update(device="gpu", gpu_platform_id=0, gpu_device_id=0)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        tr_set = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_features)
        va_set = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_features)
        try:
            model = lgb.train(gpu_params, tr_set, num_boost_round=num_boost_round,
                               valid_sets=[va_set],
                               callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        except lgb.basic.LightGBMError as e:
            if gpu_params.get("device") == "gpu":
                print(f"  [LGB] GPU 실패({e}) -> CPU로 재시도")
                model = lgb.train(params, tr_set, num_boost_round=num_boost_round,
                                   valid_sets=[va_set],
                                   callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            else:
                raise
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        models.append(model)
        if verbose_fold:
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
    gpu_params = dict(params)
    if USE_GPU_XGB:
        # xgboost>=2.0 기준 문법. tree_method는 "hist" 유지하고 device만 cuda로.
        # (예전 버전이면 tree_method="gpu_hist" 로 대체)
        gpu_params.update(device="cuda")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        try:
            model = xgb.train(gpu_params, dtr, num_boost_round=4000,
                               evals=[(dva, "valid")],
                               early_stopping_rounds=200, verbose_eval=False)
        except xgb.core.XGBoostError as e:
            if gpu_params.get("device") == "cuda":
                print(f"  [XGB] GPU 실패({e}) -> CPU로 재시도")
                model = xgb.train(params, dtr, num_boost_round=4000,
                                   evals=[(dva, "valid")],
                                   early_stopping_rounds=200, verbose_eval=False)
            else:
                raise
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
        cat_params = dict(
            iterations=4000, learning_rate=0.03, depth=8,
            l2_leaf_reg=5.0, loss_function="Logloss",
            eval_metric="Logloss", random_seed=seed,
            early_stopping_rounds=200, verbose=False,
        )
        gpu_cat_params = dict(cat_params)
        if USE_GPU_CAT:
            gpu_cat_params.update(task_type="GPU", devices="0")
        try:
            model = CatBoostClassifier(**gpu_cat_params)
            model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        except Exception as e:  # catboost는 GPU 실패 시 다양한 예외를 던질 수 있음
            if gpu_cat_params.get("task_type") == "GPU":
                print(f"  [CAT] GPU 실패({e}) -> CPU로 재시도")
                model = CatBoostClassifier(**cat_params)
                model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            else:
                raise
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(model)
        print(f"  [CAT fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


def train_rf(X, y, cat_features, n_folds=N_FOLDS, seed=SEED):
    """RandomForest는 LGB/XGB/CAT과 달리 NaN과 category dtype을 직접 못 받는다.
    - 결측치: 수치형은 중앙값 대치, 대치 여부 자체를 이미 build_features의
      *_isna 플래그 컬럼으로 갖고 있으므로 정보 손실은 제한적.
    - 범주형: 트리 기반이라 순서정보가 필요없는 원-핫 인코딩 사용.
    - GPU 미지원(sklearn 기본), n_jobs=-1로 CPU 병렬화만 적용.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    num_cols = [c for c in X.columns if c not in cat_features]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])

    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("prep", preprocessor),
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=14, min_samples_leaf=50,
                max_features="sqrt", n_jobs=-1, random_state=seed,
            )),
        ])
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(pipe)
        print(f"  [RF fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


def train_lr(X, y, cat_features, n_folds=N_FOLDS, seed=SEED):
    """LogisticRegression은 RF와 마찬가지로 NaN/category dtype을 직접 못 받는다.
    - 결측치: 수치형은 중앙값 대치, 범주형은 최빈값 대치.
    - 범주형: 원-핫 인코딩(트리와 달리 선형모델이라 순서정보 없이도 문제없음).
    - 수치형: StandardScaler로 스케일링 -> 계수 기반 최적화(lbfgs) 수렴 안정성 확보.
    - 트리/배깅 계열(LGB/XGB/CAT/RF)과 달리 선형·가법적 모델이라 상호작용을
      직접 못 잡는 대신, 구조적으로 다른 종류의 오차를 만들어 앙상블 다양성에
      기여할 가능성이 있다(RF보다 상관관계가 더 낮게 나오는지가 핵심 확인 포인트).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    num_cols = [c for c in X.columns if c not in cat_features]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])

    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        pipe = Pipeline([
            ("prep", preprocessor),
            ("lr", LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(pipe)
        print(f"  [LR fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
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

    print("Build trackman prior (USE_TRACKMAN=True일 때만 시도)...")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    prior_table = None
    if USE_TRACKMAN and os.path.exists(trackman_path):
        # 학습 데이터에 여러 시즌이 섞여 있다면, 시즌별로 컷오프를 두고
        # prior를 다시 계산하는 것이 정석이지만(진짜 leak-free),
        # 여기서는 단순화를 위해 전체 이력을 하나의 "커리어 prior"로 사용한다.
        # -> 실제 대회에서는 season 컬럼 기준으로 asof 방식으로 정교화 권장.
        prior_table = build_trackman_pitcher_prior(trackman_path)
    else:
        print("  -> 스킵 (pitcher_id 매칭률 0% 확인됨, asof_* 컬럼만 사용)")

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