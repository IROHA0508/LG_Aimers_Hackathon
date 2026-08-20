# trackman_team_feature_check.py
# ------------------------------------------------------------
# README/train_ensemble.py는 pitcher_id<->pitcher_trackman_id 직접 조인이
# 매칭률 0%라 트랙맨을 포기했었다. 그런데 별도 분석(memory: project-trackman-
# team-mapping)에서 pitcher_hand 좌투수비율 핑거프린팅으로 pitcher_team_id
# (train.csv) <-> pitcher_team 코드(trackman_history.csv)가 사실상 유일하게
# 매칭됨을 확인했다. 이 매핑을 이용해 "팀 단위" 트랙맨 프라이어(과거 시즌
# 팀 투수진의 평균 구속/회전수/무브먼트/구종비율)를 만들어 붙여본다.
#
# 검증은 지금까지 신뢰할 수 있다고 확인한 cross-regime 방식(2019-2023 학습
# -> 2024 홀드아웃)을 그대로 쓴다. 트랙맨 cutoff_season=2023으로 둬서(fit
# 기간 마지막 시즌까지만) 실전에서 2024까지 학습 -> 2025 예측 시 트랙맨
# cutoff=2024를 쓰는 것과 동일한 논리(미래 정보 사용 안 함)를 재현한다.
#
# baseline(트랙맨 없음) 대비 holdout score 460.2 (time_holdout_check.py 결과)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
BASELINE_BRIER = 0.249446

TEAM_ID_TO_TRACKMAN = {
    12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA", 16: "KIA_TIG",
    17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN", 20: "KT_WIZ", 21: "SK_WYV",
}


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def build_trackman_team_prior(trackman_path, cutoff_season, team_id_map):
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig")
    tm = tm[tm["season"] <= cutoff_season].copy()
    tm["team_merged"] = tm["pitcher_team"].replace({"SSG_LAN": "SK_WYV"})

    grp = tm.groupby("team_merged")
    prior = grp.agg(
        tm_team_n=("pitch_type_group", "size"),
        tm_team_rel_speed_mean=("rel_speed", "mean"),
        tm_team_spin_rate_mean=("spin_rate", "mean"),
        tm_team_ivb_mean=("induced_vert_break", "mean"),
        tm_team_hb_mean=("horz_break", "mean"),
        tm_team_extension_mean=("extension", "mean"),
        tm_team_rel_speed_std=("rel_speed", "std"),
    )
    mix = (tm.groupby(["team_merged", "pitch_type_group"])
             .size().unstack(fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0).add_prefix("tm_team_mix_")
    prior = prior.join(mix)

    code_to_id = {v: k for k, v in team_id_map.items()}
    prior = prior.reset_index()
    prior["pitcher_team_id"] = prior["team_merged"].map(code_to_id)
    prior = prior.dropna(subset=["pitcher_team_id"])
    prior = prior.drop(columns=["team_merged"])
    return prior


def run(use_trackman, fit_df, holdout_df, trackman_path):
    fit_df = fit_df.copy()
    holdout_df = holdout_df.copy()

    if use_trackman:
        prior = build_trackman_team_prior(trackman_path, cutoff_season=2023,
                                           team_id_map=TEAM_ID_TO_TRACKMAN)
        print(f"  trackman team prior rows: {len(prior)} (of {len(TEAM_ID_TO_TRACKMAN)} mapped teams)")
        fit_df = fit_df.merge(prior, on="pitcher_team_id", how="left")
        holdout_df = holdout_df.merge(prior, on="pitcher_team_id", how="left")
        match_rate = fit_df["tm_team_n"].notna().mean()
        print(f"  fit match rate (team-level): {match_rate:.1%}")

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
    print(f"  n_features={len(feat_cols)}")

    models, oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y_fit)

    holdout_pred = np.mean(
        [m.predict(X_holdout, num_iteration=m.best_iteration) for m in models], axis=0)
    holdout_brier = brier_score_loss(y_holdout, holdout_pred)
    holdout_cal_brier = brier_score_loss(y_holdout, iso.predict(holdout_pred))

    # feature importance check for trackman columns (있으면)
    if use_trackman:
        imp = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
        imp_df = pd.DataFrame({"feature": feat_cols, "gain": imp}).sort_values("gain", ascending=False)
        tm_cols = [c for c in feat_cols if c.startswith("tm_team_")]
        print("\n  trackman team feature importance (gain, rank / total):")
        imp_df = imp_df.reset_index(drop=True)
        for c in tm_cols:
            rank = imp_df.index[imp_df["feature"] == c][0] + 1
            gain = imp_df.loc[imp_df["feature"] == c, "gain"].values[0]
            print(f"    {c:30s} rank={rank}/{len(feat_cols)}  gain={gain:.1f}")

    return score(holdout_brier), score(holdout_cal_brier)


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    trackman_path = f"{DATA_DIR}/trackman_history.csv"

    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()
    print(f"fit={len(fit_df)}  holdout={len(holdout_df)}")

    print("\n=== baseline (no trackman) ===")
    base_raw, base_cal = run(False, fit_df, holdout_df, trackman_path)
    print(f"  holdout score raw={base_raw:.1f} calibrated={base_cal:.1f}")

    print("\n=== with trackman team-level features ===")
    tm_raw, tm_cal = run(True, fit_df, holdout_df, trackman_path)
    print(f"  holdout score raw={tm_raw:.1f} calibrated={tm_cal:.1f}")

    print("\n\n=== SUMMARY ===")
    print(f"baseline (no trackman):       raw={base_raw:.1f}  calibrated={base_cal:.1f}")
    print(f"with trackman team features:  raw={tm_raw:.1f}  calibrated={tm_cal:.1f}")
    print(f"delta (calibrated): {tm_cal - base_cal:+.1f}")


if __name__ == "__main__":
    main()
