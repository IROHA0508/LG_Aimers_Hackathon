# script.py
# ------------------------------------------------------------
# 투구 제구 성공 확률 예측 - 추론 전용 스크립트 (대회 평가 서버에서 실행).
# 학습 코드는 없음 - model/ensemble_bundle.pkl에 저장된 LightGBM(튜닝됨) +
# Entity-Embedding MLP 2모델 스태킹(Isotonic 보정 -> Logistic 메타러너)을
# 그대로 불러와 test.csv에 적용한다. 학습 파이프라인은 code/train_ensemble.py
# (제출 대상 아님, 대회 10분 추론 제한과 무관).
#
# data/ 에서 test.csv/sample_submission.csv를 읽고 output/submission.csv를 쓴다.
# 인터넷 접속 없이 완전히 오프라인으로 동작한다 (모든 가중치/lookup 테이블은
# model/ensemble_bundle.pkl 안에 이미 고정돼 있음).
# ------------------------------------------------------------
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ID_COL = "row_id"
TARGET_COL = "control_success"

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


# =======================
# Entity-Embedding MLP (code/train_ensemble.py의 EntityEmbedMLP와 반드시 동일해야
# state_dict가 그대로 로드된다 - 구조를 바꾸려면 학습도 다시 해야 함)
# =======================

class EntityEmbedMLP(nn.Module):
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
    cat_arr = np.zeros((len(X), len(cat_features)), dtype=np.int64)
    for i, c in enumerate(cat_features):
        uniques = cat_maps[c]
        code_map = {v: i2 for i2, v in enumerate(uniques)}
        cat_arr[:, i] = X[c].astype(str).map(code_map).fillna(len(uniques)).astype(np.int64)
    num_arr = num_imputer.transform(X[num_cols])
    num_arr = scaler.transform(num_arr)
    return cat_arr, num_arr


def predict_mlp(state_dicts, preprocessor, X):
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


# =======================
# 피처 생성 (추론 전용 - code/train_ensemble.py의 build_features와 동일한 파생
# 로직을 쓰되, test.csv엔 라벨이 없으므로 학습 시 저장해둔 snapshot_tables/
# cat_dtype_categories/trackman_prior_table 고정값을 조인하는 경로만 사용한다.
# 학습 시점 누적치 계산(cumcount/cumsum) 코드는 여기 없음 - 순수 추론만 수행)
# =======================

def build_features_infer(df, bundle):
    snapshot_tables = bundle.get("snapshot_tables") or {}
    trackman_prior = bundle.get("trackman_prior_table")
    cat_dtype_categories = bundle["cat_dtype_categories"]

    df = df.copy()

    # 팀 as-of (학습 종료 시점 최종 스냅샷 조인)
    for team_col, prefix in [("pitcher_team_id", "pitcher_team"),
                              ("batter_team_id", "batter_team")]:
        if team_col in df.columns and prefix in snapshot_tables:
            df = df.merge(snapshot_tables[prefix], on=team_col, how="left")

    # 팀x팀 맞대결 as-of
    if "team_matchup" in snapshot_tables and {"pitcher_team_id", "batter_team_id"}.issubset(df.columns):
        df = df.merge(snapshot_tables["team_matchup"],
                       on=["pitcher_team_id", "batter_team_id"], how="left")

    # 시즌 추세 (train에 없는 미래 시즌은 가장 최근 시즌 값으로 대체)
    if "season" in snapshot_tables and "season" in df.columns:
        df = df.merge(snapshot_tables["season"], on="season", how="left")
        df["asof_season_success_rate"] = df["asof_season_success_rate"].fillna(
            snapshot_tables["latest_season_rate"])
        if "asof_pitcher_success_rate" in df.columns:
            df["pitcher_rate_vs_season"] = df["asof_pitcher_success_rate"] - df["asof_season_success_rate"]
        if "asof_batter_success_rate" in df.columns:
            df["batter_rate_vs_season"] = df["asof_batter_success_rate"] - df["asof_season_success_rate"]

    # cold-start(표본 0) 결측 플래그
    for c in RATE_COLS_FOR_MISSING_FLAG:
        if c in df.columns:
            df[f"{c}_isna"] = df[c].isna().astype(np.int8)

    # 표본 수 기반 신뢰도 가중
    if "asof_pitcher_n" in df.columns:
        df["asof_pitcher_n_log"] = np.log1p(df["asof_pitcher_n"])
    if "asof_batter_n" in df.columns:
        df["asof_batter_n_log"] = np.log1p(df["asof_batter_n"])
    if "asof_pitcher_pitchmix_n" in df.columns:
        df["asof_pitcher_pitchmix_n_log"] = np.log1p(df["asof_pitcher_pitchmix_n"])

    # 카운트 파생
    if {"balls_before", "strikes_before"}.issubset(df.columns):
        df["count_pressure"] = df["strikes_before"] - df["balls_before"]
        df["is_two_strike"] = (df["strikes_before"] == 2).astype(np.int8)
        df["is_three_ball"] = (df["balls_before"] == 3).astype(np.int8)

    # 좌우 상성(platoon split)
    if {"pitcher_hand", "batter_hand"}.issubset(df.columns):
        df["is_same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int8)
        df["hand_matchup"] = (
            df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
        )

    # 트랙맨 커리어 prior 조인 (USE_TRACKMAN=False라 실제로는 None -> 스킵)
    if trackman_prior is not None:
        df = df.merge(trackman_prior, on="pitcher_id", how="left")

    # 범주형 -> 학습 시 카테고리 그대로 강제 적용 (코드 불일치 방지)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=cat_dtype_categories[c])

    return df


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 ----
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_bundle.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 번들 로드 ----
    print("Load model bundle...")
    bundle = joblib.load(MODEL_PATH)
    print(f" OK. n_features={len(bundle['feat_cols'])}, "
          f"lgb_folds={len(bundle['lgb_models'])}, mlp_folds={len(bundle['mlp_state_dicts'])}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 피처 생성 (학습과 동일한 파생 로직, 추론 전용 경로) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    test_feat = build_features_infer(test, bundle)
    X = test_feat[bundle["feat_cols"]]
    print(f" features={X.shape[1]} rows={len(X)}")

    if len(X) == 0:
        preds = []
    else:
        # ---- LightGBM 추론 (5-fold 평균) ----
        print("Inference LightGBM...")
        lgb_preds = np.mean(
            [m.predict(X, num_iteration=m.best_iteration) for m in bundle["lgb_models"]], axis=0)

        # ---- MLP 추론 (5-fold 평균) ----
        print("Inference MLP...")
        mlp_preds = predict_mlp(bundle["mlp_state_dicts"], bundle["mlp_preprocessor"], X)

        # ---- 보정 + 스태킹 ----
        print("Calibrate + stack...")
        lgb_c = bundle["calibrators"]["lgb"].predict(lgb_preds)
        mlp_c = bundle["calibrators"]["mlp"].predict(mlp_preds)
        meta_X = np.column_stack([lgb_c, mlp_c])
        stack_pred = bundle["meta_model"].predict_proba(meta_X)[:, 1]
        preds = bundle["final_calibrator"].predict(stack_pred)

    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
