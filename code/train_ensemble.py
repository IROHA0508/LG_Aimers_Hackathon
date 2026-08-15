# train_ensemble.py
# ------------------------------------------------------------
# 투구 제구 성공 확률 예측 - 학습 파이프라인
# 전략(v2, 최종 채택): LightGBM(튜닝됨) + Entity-Embedding MLP(딥러닝) 2모델 앙상블
#       -> OOF 예측 생성 -> Isotonic 확률보정 -> Logistic 메타러너 스태킹
# 평가지표(Brier Skill Score)는 판별력보다 "보정 품질"에 민감하므로
# 보정 단계를 파이프라인 필수 스텝으로 둔다.
#
# final_check.py / rf_check.py / lr_check.py / mlp_check.py / full_stack_check.py로
# 전체 데이터 기준 비교한 결과(2026-08-15~16, 상세 내역은 README.md 참고):
#   LightGBM 단독              : score=2397.8
#   LGB+XGB+CAT 3모델 스태킹   : score=2381.9  (LGB 단독보다 -15.9, 희석효과)
#   LGB+RF 2모델 스태킹        : score=2409.2  (LGB 단독보다 +11.4)
#   LGB+RF+LR 3모델 스태킹     : score=2394.6  (2way보다 -14.5, LR이 다양성도 크지 않은데 성능만 나빠 희석)
#   LGB+MLP 2모델 스태킹       : score=2411.4  (LGB 단독보다 +13.7, 지금까지 최고점 -> 최종 채택)
#   LGB+RF+MLP 3모델 스태킹    : score=2411.6  (LGB+MLP보다 +0.1, RF-MLP 상관계수 0.901로 중복 -> 기각)
# 핵심 교훈: 스태킹 이득은 "단독 성능"이 아니라 "LGB와의 오차 상관관계"가 결정한다.
# XGB/CAT은 LGB와 같은 부스팅 계열이라 상관관계가 높아(추정) 손해였고, RF/MLP는
# 배깅/신경망 기반이라 상관관계가 낮아(0.94/0.87) 단독 성능이 약해도 도움이 됐다.
# MLP가 RF보다 상관관계가 더 낮아(0.866 vs 0.939) 최종 점수가 더 높았다.
# 이에 따라 XGB/CAT/RF는 메인 파이프라인에서 제외하고 LGB+MLP 2모델 스태킹을
# 최종 아키텍처로 채택한다. (train_xgb/train_cat/train_rf/train_lr 함수 자체는
# 튜닝·실험 근거 보존을 위해 남겨둠)
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# xgboost/catboost는 최종 아키텍처(LGB+MLP)에서 빠졌으므로 모듈 최상단에서 임포트하지
# 않는다 - train_xgb/train_cat 함수 내부에서만 지연 임포트한다. 이렇게 해야
# script.py가 이 모듈을 그대로 import해도 xgboost/catboost 설치 없이 동작한다
# (제출 시 requirements.txt에서 불필요한 패키지를 뺄 수 있음).

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


def build_train_snapshot_tables(train_df):
    """train.csv 전체를 다 본 "최종 스냅샷" lookup 테이블 생성 (추론 전용).

    build_team_features 등은 학습 시(TARGET_COL 존재)엔 시점별 누적 as-of 값을
    쓰지만, 추론 시(test.csv)엔 라벨이 없어 누적을 이어갈 방법이 없다. 대신
    train.csv를 끝까지 다 본 "최종 상태"를 새 행(test)에 고정 피처로 붙인다.
    trackman_prior_table과 동일한 논리 - 과거 전체 이력을 미래 데이터의 고정
    prior로 제공할 뿐 미래 정보를 쓰지 않으므로 리크가 아니다.
    """
    tables = {}
    for team_col, prefix in [("pitcher_team_id", "pitcher_team"),
                              ("batter_team_id", "batter_team")]:
        if team_col not in train_df.columns:
            continue
        g = train_df.groupby(team_col)[TARGET_COL].agg(n="count", success="sum").reset_index()
        g[f"asof_{prefix}_n"] = g["n"]
        g[f"asof_{prefix}_success_rate"] = g["success"] / g["n"]
        tables[prefix] = g[[team_col, f"asof_{prefix}_n", f"asof_{prefix}_success_rate"]]

    if {"pitcher_team_id", "batter_team_id"}.issubset(train_df.columns):
        g = (train_df.groupby(["pitcher_team_id", "batter_team_id"])[TARGET_COL]
             .agg(n="count", success="sum").reset_index())
        g["asof_team_matchup_n"] = g["n"]
        g["asof_team_matchup_success_rate"] = g["success"] / g["n"]
        tables["team_matchup"] = g[["pitcher_team_id", "batter_team_id",
                                     "asof_team_matchup_n", "asof_team_matchup_success_rate"]]

    if "season" in train_df.columns:
        g = train_df.groupby("season")[TARGET_COL].agg(n="count", success="sum").reset_index()
        g["asof_season_success_rate"] = g["success"] / g["n"]
        tables["season"] = g[["season", "asof_season_success_rate"]]
        # test.csv 시즌(예: 2025)이 train에 없는 미래 시즌이면 가장 최근 시즌 값으로 대체
        tables["latest_season_rate"] = g.sort_values("season")["asof_season_success_rate"].iloc[-1]

    return tables


def build_team_features(df, snapshot_tables=None):
    """팀 단위 as-of 피처 (leak-free).

    선수 개인 단위(asof_pitcher_*, asof_batter_*)와 별개로 팀 전체 단위의
    과거 경향(코칭 스타일, 구장 요인, 포수진 등 개인 통계로는 못 잡는
    조직 단위 신호)을 노려본다. 팀 ID는 13개뿐이라 표본이 매우 커서
    (팀당 평균 11만행) 콜드스타트 영향은 거의 없다고 봐도 된다.

    pitcher_team_id는 시즌 중 이적으로 바뀔 수 있음을 확인했다(선수의 20.7%가
    커리어 중 팀ID 2개 이상). team_id 기준으로 그룹을 나누므로 이적 시점에
    자동으로 그 팀의 누적치로 전환되어 별도 처리가 필요 없다.

    matchup 피처와 동일하게 (그룹 내 "현재 행 이전"만 사용) 누수를 방지한다.
    TARGET_COL이 없으면(추론) snapshot_tables(build_train_snapshot_tables)의
    고정값을 조인한다.
    """
    if TARGET_COL in df.columns:
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

    if not snapshot_tables:
        return df
    for team_col, prefix in [("pitcher_team_id", "pitcher_team"),
                              ("batter_team_id", "batter_team")]:
        if team_col not in df.columns or prefix not in snapshot_tables:
            continue
        df = df.merge(snapshot_tables[prefix], on=team_col, how="left")
    return df


def build_team_matchup_features(df, snapshot_tables=None):
    """팀x팀(투수팀 vs 타자팀) as-of 맞대결 피처.

    투수-타자 개별 맞대결(build_matchup_features)은 조합이 96,133개라
    조합당 평균 15행뿐이라 실패했지만, 팀 단위로 묶으면 조합이 96개뿐이라
    조합당 평균 15,365행(최소 292행)으로 표본 규모가 완전히 다르다.
    team 단위 피처(build_team_features)가 성공했던 것과 같은 이유로
    성공 가능성을 기대해볼 수 있다.

    TARGET_COL이 없으면(추론) snapshot_tables의 고정값을 조인한다.
    """
    if TARGET_COL in df.columns:
        df = df.sort_values(ID_COL).reset_index(drop=True)
        grp_key = ["pitcher_team_id", "batter_team_id"]
        n = df.groupby(grp_key).cumcount()
        cum_success = df.groupby(grp_key)[TARGET_COL].cumsum() - df[TARGET_COL]
        df["asof_team_matchup_n"] = n
        df["asof_team_matchup_success_rate"] = cum_success / n.replace(0, np.nan)
        return df

    if not snapshot_tables or "team_matchup" not in snapshot_tables:
        return df
    df = df.merge(snapshot_tables["team_matchup"],
                   on=["pitcher_team_id", "batter_team_id"], how="left")
    return df


def build_season_trend_features(df, snapshot_tables=None):
    """시즌 기준선 대비 상대 성공률 (era-adjusted rate).

    실측 결과 시즌별 control_success 비율이 2019 0.565 -> 2024 0.486로
    6년간 7.9%p 꾸준히 하락하는 큰 추세가 있다. asof_pitcher/batter_success_rate는
    커리어 누적 평균이라 이 추세가 섞여 있어(예: 2019~2024를 다 뛴 베테랑은
    성공률 높았던 초기 시즌과 낮아진 최근 시즌이 뭉개진 평균), "이 선수가 리그
    평균 대비 실제로 잘하는지"를 흐릴 수 있다. 시즌 자체의 as-of 기준선을
    별도로 만들어 빼주면 선수 개인의 순수 편차를 더 선명하게 분리할 수 있다.

    시즌 그룹 크기가 시즌당 평균 24만행으로 매우 커서 콜드스타트(시즌 첫 몇 행)는
    무시할 수준이다.

    TARGET_COL이 없으면(추론) snapshot_tables의 시즌별 최종 성공률을 조인한다.
    test.csv의 시즌(예: 2025)이 train에 없는 미래 시즌이면 train의 가장 최근
    시즌 값(latest_season_rate)으로 대체한다 - 추세가 계속될 거란 가정의 근사치.
    """
    if TARGET_COL in df.columns:
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

    if not snapshot_tables or "season" not in snapshot_tables:
        return df
    df = df.merge(snapshot_tables["season"], on="season", how="left")
    df["asof_season_success_rate"] = df["asof_season_success_rate"].fillna(
        snapshot_tables["latest_season_rate"])
    if "asof_pitcher_success_rate" in df.columns:
        df["pitcher_rate_vs_season"] = df["asof_pitcher_success_rate"] - df["asof_season_success_rate"]
    if "asof_batter_success_rate" in df.columns:
        df["batter_rate_vs_season"] = df["asof_batter_success_rate"] - df["asof_season_success_rate"]
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
                    use_season_trend_feature=True,
                    snapshot_tables=None, cat_dtype_categories=None):
    # snapshot_tables: build_train_snapshot_tables(train_df)의 결과. 추론 시(df에
    # TARGET_COL이 없을 때) 팀/팀맞대결/시즌 as-of 피처를 학습 시점 최종 스냅샷
    # 고정값으로 채우기 위해 필요 (없으면 이 피처들이 통째로 누락됨).
    # cat_dtype_categories: 학습 시 {컬럼명: 카테고리 목록} 저장값. 추론 시 pandas
    # category dtype이 학습 때와 다른 코드로 매겨지는 걸 방지하기 위해 강제 적용
    # (LightGBM은 카테고리의 문자열이 아니라 내부 정수 코드로 분기를 저장하므로,
    # 코드가 어긋나면 에러 없이 조용히 틀린 예측을 낸다).
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
        df = build_team_features(df, snapshot_tables=snapshot_tables)
    if use_clutch_feature:
        df = build_clutch_features(df)
    if use_team_matchup_feature:
        df = build_team_matchup_features(df, snapshot_tables=snapshot_tables)
    if use_season_trend_feature:
        df = build_season_trend_features(df, snapshot_tables=snapshot_tables)

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
    # cat_dtype_categories가 주어지면(추론 시) 학습 때의 카테고리 목록/순서를
    # 그대로 강제 적용한다. LightGBM은 카테고리를 내부 정수 코드로 저장하므로,
    # df마다 독립적으로 astype("category")하면 코드가 어긋나 조용히 오예측할 수 있다.
    for c in CAT_COLS:
        if c not in df.columns:
            continue
        if cat_dtype_categories is not None and c in cat_dtype_categories:
            df[c] = pd.Categorical(df[c], categories=cat_dtype_categories[c])
        else:
            df[c] = df[c].astype("category")

    drop_cols = [ID_COL, TARGET_COL]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return df, feat_cols


# ============================================================
# 3) 모델별 학습 함수 (OOF 반환)
# ============================================================

def train_lgb(X, y, X_full, cat_features, n_folds=N_FOLDS, seed=SEED,
              params_override=None, early_stopping_rounds=300, verbose_fold=True,
              num_boost_round=8000):
    # 하이퍼파라미터는 grid_search_lgb.py/_round2/_round3의 3단계 순차 그리드서치로
    # 튜닝됨 (2-fold 탐색 -> 5-fold 최종검증). 전체 데이터 기준 Brier 개선:
    # 기본값(튜닝 전) 0.243759 -> 1차 0.243624 -> 2차 0.243497 -> 3차 0.243454
    # (점수 환산 +122.2점, 이번 세션 전체에서 가장 큰 단일 개선폭). 3차에서 개선폭이
    # 1/3로 줄어들어(수확체감 + 117개 조합이 같은 2-fold에 과적합될 위험) 3차에서 중단.
    # learning_rate가 낮아진 만큼 num_boost_round/early_stopping_rounds도 검증에 쓴
    # 값(8000/300)으로 같이 올림 - 그래야 실제 학습 시에도 충분히 수렴함.
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.005, num_leaves=320, min_data_in_leaf=800,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=10.0, max_depth=-1, verbose=-1, seed=seed,
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


def train_xgb(X, y, cat_features, n_folds=N_FOLDS, seed=SEED,
              params_override=None, early_stopping_rounds=200, verbose_fold=True,
              num_boost_round=4000):
    import xgboost as xgb
    # XGBoost native categorical 지원 (enable_categorical=True)
    # 하이퍼파라미터는 grid_search_xgb.py의 2단계 순차 그리드서치로 튜닝됨
    # (2-fold 탐색 -> 5-fold 최종검증). 전체 데이터 기준 Brier: 기본값 0.243809 ->
    # 튜닝값 0.243708 (점수 환산 +40.5점). max_depth/min_child_weight/colsample_bytree/
    # reg_lambda는 탐색 경계값으로 나왔으나, CAT이 기본값도 오래 걸려 스킵하기로
    # 한 것과 같은 이유로 추가 확장 라운드는 진행하지 않음(시간 대비 수확체감 판단).
    oof = np.zeros(len(X))
    models = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    params = dict(
        objective="binary:logistic", eval_metric="logloss",
        eta=0.02, max_depth=10, min_child_weight=40,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0,
        tree_method="hist", enable_categorical=True, seed=seed,
    )
    if params_override:
        params.update(params_override)
    gpu_params = dict(params)
    if USE_GPU_XGB:
        # xgboost>=2.0 기준 문법. tree_method는 "hist" 유지하고 device만 cuda로.
        # (예전 버전이면 tree_method="gpu_hist" 로 대체)
        gpu_params.update(device="cuda")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        try:
            model = xgb.train(gpu_params, dtr, num_boost_round=num_boost_round,
                               evals=[(dva, "valid")],
                               early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        except xgb.core.XGBoostError as e:
            if gpu_params.get("device") == "cuda":
                print(f"  [XGB] GPU 실패({e}) -> CPU로 재시도")
                model = xgb.train(params, dtr, num_boost_round=num_boost_round,
                                   evals=[(dva, "valid")],
                                   early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            else:
                raise
        oof[va_idx] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        models.append(model)
        if verbose_fold:
            print(f"  [XGB fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f}")
    return models, oof


def train_cat(X, y, cat_features, n_folds=N_FOLDS, seed=SEED):
    from catboost import CatBoostClassifier, Pool
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


class EntityEmbedMLP(nn.Module):
    """범주형은 임베딩, 수치형은 concat -> MLP(BatchNorm+Dropout) -> logit.
    모듈 레벨 클래스로 둬야 한다 - train_mlp 내부 지역 클래스로 두면 joblib/pickle이
    학습 프로세스 밖(예: script.py)에서 그 클래스 정의를 못 찾아 모델 복원에 실패한다.
    추론 시엔 이 클래스 정의 + 저장된 state_dict로 모델을 재구성한다(가중치만 저장,
    표준 PyTorch 배포 방식).
    """
    def __init__(self, cat_cardinalities, n_num, hidden_dims, dropout, embed_dim_cap):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(card, min(embed_dim_cap, max(2, (card + 1) // 2)))
            for card in cat_cardinalities.values()
        ])
        in_dim = sum(e.embedding_dim for e in self.embeds) + n_num
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        return self.mlp(torch.cat(embs + [x_num], dim=1)).squeeze(1)


def encode_mlp_inputs(X, cat_features, num_cols, cat_maps, num_imputer, scaler):
    """train_mlp이 학습 시 fit한 전처리(cat_maps/num_imputer/scaler)를 그대로
    적용해 (cat_codes, num_array)를 만든다. 학습/추론 공용 - 추론 시 미지 카테고리는
    자동으로 "unknown" 인덱스(len(uniques))로 몰린다(cat_maps에 저장된 매핑 기준).
    """
    cat_arr = np.zeros((len(X), len(cat_features)), dtype=np.int64)
    for i, c in enumerate(cat_features):
        uniques = cat_maps[c]
        code_map = {v: i2 for i2, v in enumerate(uniques)}
        cat_arr[:, i] = X[c].astype(str).map(code_map).fillna(len(uniques)).astype(np.int64)
    num_arr = num_imputer.transform(X[num_cols])
    num_arr = scaler.transform(num_arr)
    return cat_arr, num_arr


def train_mlp(X, y, cat_features, n_folds=N_FOLDS, seed=SEED,
              embed_dim_cap=16, hidden_dims=(256, 128), dropout=0.3,
              lr=1e-3, batch_size=4096, max_epochs=30, patience=5,
              verbose_fold=True):
    """Entity-Embedding MLP. torch만 필요(GPU 자동 사용). 참고: 다수 벤치마크
    (Grinsztajn et al. 2022 등)에서 GBDT가 정형 데이터 단독 성능은 앞서는 경향이
    확인돼, 이 모델은 LGB를 이길 목적이 아니라 스태킹 다양성 후보로 검증됐다
    (lgb-mlp 상관계수 0.866 < lgb-rf 0.939, RF보다 더 큰 스태킹 이득 확인 -> 최종 채택).
    검증 폴드 brier가 patience 에폭 연속 개선 없으면 조기종료, best 가중치 복원.

    전처리(cat_maps/num_imputer/scaler)는 폴드가 아니라 전체 X 기준으로 한 번만
    fit한다 - 폴드마다 따로 fit하면 추론 시 어떤 전처리를 재사용해야 할지 알 수
    없게 되는 문제(초기 버전의 버그)를 피하기 위함. 1.47M행 규모에서 폴드 subset과
    전체의 중앙값/평균 차이는 무시할 수준이라 성능 영향은 없다.

    반환값의 models는 nn.Module 인스턴스가 아니라 state_dict 리스트다(표준 PyTorch
    배포 방식 - pickle이 아니라 텐서 딕셔너리라 어디서든 EntityEmbedMLP 클래스만
    있으면 복원 가능).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cols = [c for c in X.columns if c not in cat_features]

    num_imputer = SimpleImputer(strategy="median").fit(X[num_cols])
    scaler = StandardScaler().fit(num_imputer.transform(X[num_cols]))

    cat_maps = {}
    cat_cardinalities = {}
    cat_codes = pd.DataFrame(index=X.index)
    for c in cat_features:
        codes, uniques = pd.factorize(X[c].astype(str))
        codes = np.where(codes == -1, len(uniques), codes)
        cat_codes[c] = codes
        cat_maps[c] = list(uniques)
        cat_cardinalities[c] = len(uniques) + 1

    num_all = scaler.transform(num_imputer.transform(X[num_cols]))

    oof = np.zeros(len(X))
    state_dicts = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        cat_tr, cat_va = cat_codes.iloc[tr_idx].values, cat_codes.iloc[va_idx].values
        num_tr, num_va = num_all[tr_idx], num_all[va_idx]

        tr_ds = TensorDataset(
            torch.tensor(cat_tr, dtype=torch.long),
            torch.tensor(num_tr, dtype=torch.float32),
            torch.tensor(y[tr_idx], dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

        cat_va_t = torch.tensor(cat_va, dtype=torch.long, device=device)
        num_va_t = torch.tensor(num_va, dtype=torch.float32, device=device)

        model = EntityEmbedMLP(cat_cardinalities, len(num_cols), hidden_dims, dropout, embed_dim_cap).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        best_brier = np.inf
        best_state = None
        no_improve = 0
        for epoch in range(max_epochs):
            model.train()
            for xb_cat, xb_num, yb in tr_loader:
                xb_cat, xb_num, yb = xb_cat.to(device), xb_num.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb_cat, xb_num), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                va_pred = torch.sigmoid(model(cat_va_t, num_va_t)).cpu().numpy()
            va_brier = brier_score_loss(y[va_idx], va_pred)
            if va_brier < best_brier - 1e-6:
                best_brier = va_brier
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(model(cat_va_t, num_va_t)).cpu().numpy()
        state_dicts.append(best_state)
        if verbose_fold:
            print(f"  [MLP fold {fold}] brier={brier_score_loss(y[va_idx], oof[va_idx]):.5f} (epoch={epoch+1})")

    preprocessor = {
        "num_cols": num_cols, "cat_features": cat_features,
        "num_imputer": num_imputer, "scaler": scaler,
        "cat_maps": cat_maps, "cat_cardinalities": cat_cardinalities,
        "hidden_dims": hidden_dims, "dropout": dropout, "embed_dim_cap": embed_dim_cap,
    }
    return state_dicts, oof, preprocessor


def predict_mlp(state_dicts, preprocessor, X):
    """train_mlp이 저장한 state_dicts+preprocessor로 X를 추론한다 (폴드 평균).
    학습 때와 동일하게 encode_mlp_inputs로 전처리 -> 각 폴드 state_dict를 로드한
    EntityEmbedMLP로 예측 -> 폴드 평균."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_arr, num_arr = encode_mlp_inputs(
        X, preprocessor["cat_features"], preprocessor["num_cols"],
        preprocessor["cat_maps"], preprocessor["num_imputer"], preprocessor["scaler"])
    cat_t = torch.tensor(cat_arr, dtype=torch.long, device=device)
    num_t = torch.tensor(num_arr, dtype=torch.float32, device=device)

    preds = np.zeros(len(X))
    model = EntityEmbedMLP(preprocessor["cat_cardinalities"], len(preprocessor["num_cols"]),
                            preprocessor["hidden_dims"], preprocessor["dropout"],
                            preprocessor["embed_dim_cap"]).to(device)
    for state_dict in state_dicts:
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()
        with torch.no_grad():
            preds += torch.sigmoid(model(cat_t, num_t)).cpu().numpy()
    return preds / len(state_dicts)


# ============================================================
# 4) 메인
# ============================================================

def main():
    DATA_DIR = "../open/data"
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

    print("Build train snapshot lookup tables (팀/팀맞대결/시즌 - 추론 시 test.csv용)...")
    snapshot_tables = build_train_snapshot_tables(train)

    print("Build features...")
    train_feat, feat_cols = build_features(train, prior_table)
    cat_features = [c for c in CAT_COLS if c in feat_cols]

    # 추론 시 test.csv에 동일한 카테고리 코드를 강제 적용하기 위해 학습 시 사용된
    # 카테고리 목록을 저장한다 (script.py에서 cat_dtype_categories로 재사용).
    cat_dtype_categories = {c: train_feat[c].cat.categories for c in cat_features}

    X = train_feat[feat_cols]
    print(f"  n_features={len(feat_cols)}, n_rows={len(X)}")

    print("Train LightGBM (튜닝됨)...")
    lgb_models, lgb_oof = train_lgb(X, y, X, cat_features)

    print("Train Entity-Embedding MLP...")
    mlp_state_dicts, mlp_oof, mlp_preprocessor = train_mlp(X, y, cat_features)

    print("Calibrate each model's OOF (Isotonic)...")
    calibrators = {}
    calibrated = {}
    for name, oof in [("lgb", lgb_oof), ("mlp", mlp_oof)]:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y)
        calibrators[name] = iso
        calibrated[name] = iso.predict(oof)
        print(f"  [{name}] raw brier={brier_score_loss(y, oof):.5f} "
              f"-> calibrated brier={brier_score_loss(y, calibrated[name]):.5f}")

    print("Stack (meta Logistic Regression on calibrated OOF probs)...")
    meta_X = np.column_stack([calibrated["lgb"], calibrated["mlp"]])
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
        "mlp_state_dicts": mlp_state_dicts,
        "mlp_preprocessor": mlp_preprocessor,
        "calibrators": calibrators,
        "meta_model": meta,
        "final_calibrator": final_iso,
        "feat_cols": feat_cols,
        "cat_features": cat_features,
        "cat_dtype_categories": cat_dtype_categories,  # 추론 시 카테고리 코드 일치용
        "snapshot_tables": snapshot_tables,  # 추론 시 팀/팀맞대결/시즌 as-of 피처 조인용
        "trackman_prior_table": prior_table,  # inference 시 원본 트랙맨 파일 불필요
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "ensemble_bundle.pkl"))
    print("[OK] Saved model/ensemble_bundle.pkl")


if __name__ == "__main__":
    main()