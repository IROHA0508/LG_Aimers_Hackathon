# trackman_pitcher_feature_check.py
# ------------------------------------------------------------
# trackman_pitcher_matching.py로 만든 pitcher_id -> pitcher_trackman_id
# 매핑(code/trackman_pitcher_map.csv)을 이용해 선수 개인 단위 트랙맨 프라이어
# (구속/회전수/무브먼트/구종비율)를 붙이고, cross-regime holdout(2019-2023
# 학습 -> 2024 검증)으로 예측력을 테스트한다.
#
# 매핑 신뢰도 두 단계를 각각 테스트한다:
#   high_conf: 2개 이상 시즌에서 100% 일치한 매칭만(62명, fit기간 row 17.5% 커버)
#              - 독립검증 상관계수 fastball=0.62, breaking=0.85로 신뢰도 높음
#   all_resolved: high_conf + 단일시즌 매칭까지 전부(290명, row 39% 커버)
#              - 단일시즌만 매칭된 건 독립검증 상관계수가 0.20~0.24로 낮아 노이즈 섞임
#
# baseline(트랙맨 없음) holdout score = 460.2 (calibrated 기준)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te

DATA_DIR = "../open/data"
TARGET_COL = te.TARGET_COL
BASELINE_BRIER = 0.249446


def score(brier):
    return max(0.0, 100000 * (1 - brier / BASELINE_BRIER))


def build_trackman_pitcher_prior_mapped(trackman_path, cutoff_season, id_map_df):
    """train_ensemble.build_trackman_pitcher_prior과 동일 로직이지만, 직접 조인
    대신 Hungarian 매칭으로 얻은 pitcher_id<->pitcher_trackman_id 매핑을 거친다."""
    tm = pd.read_csv(trackman_path, encoding="utf-8-sig")
    tm = tm[tm["season"] <= cutoff_season]

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

    # pitcher_trackman_id -> pitcher_id 매핑(우리가 만든 id_map)을 거쳐 train.csv의
    # pitcher_id 기준 prior 테이블로 변환
    prior = id_map_df[["pitcher_id", "pitcher_trackman_id"]].merge(prior, on="pitcher_trackman_id", how="left")
    prior = prior.drop(columns=["pitcher_trackman_id"])
    return prior


def run(name, id_map_df, fit_df, holdout_df, trackman_path):
    print(f"\n=== {name} ===")
    fit_df = fit_df.copy()
    holdout_df = holdout_df.copy()

    prior = None
    if id_map_df is not None:
        prior = build_trackman_pitcher_prior_mapped(trackman_path, cutoff_season=2023, id_map_df=id_map_df)
        print(f"  prior table rows: {len(prior)}")
        fit_df = fit_df.merge(prior, on="pitcher_id", how="left")
        holdout_df = holdout_df.merge(prior, on="pitcher_id", how="left")
        print(f"  fit row match rate: {fit_df['tm_n'].notna().mean():.1%}  "
              f"holdout row match rate: {holdout_df['tm_n'].notna().mean():.1%}")

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

    if id_map_df is not None:
        imp = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
        imp_df = pd.DataFrame({"feature": feat_cols, "gain": imp}).sort_values("gain", ascending=False).reset_index(drop=True)
        tm_cols = [c for c in feat_cols if c.startswith("tm_")]
        print("  trackman pitcher feature importance (gain rank / total):")
        for c in tm_cols:
            idx = imp_df.index[imp_df["feature"] == c]
            if len(idx):
                print(f"    {c:20s} rank={idx[0]+1}/{len(feat_cols)}  gain={imp_df.loc[idx[0],'gain']:.1f}")

    print(f"  holdout(raw) score={score(holdout_brier):.1f}  holdout(calibrated) score={score(holdout_cal_brier):.1f}")
    return score(holdout_brier), score(holdout_cal_brier)


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    trackman_path = f"{DATA_DIR}/trackman_history.csv"
    final_map = pd.read_csv("trackman_pitcher_map.csv")
    high_conf = final_map[(final_map["n_seasons_seen"] >= 2) & (final_map["agree_rate"] == 1.0)]

    fit_df = train[train["season"] <= 2023].copy()
    holdout_df = train[train["season"] == 2024].copy()
    print(f"fit={len(fit_df)}  holdout={len(holdout_df)}")
    print(f"high_conf mapping: {len(high_conf)} pitchers, all_resolved: {len(final_map)} pitchers")

    results = {}
    results["baseline"] = run("baseline (no trackman pitcher features)", None, fit_df, holdout_df, trackman_path)
    results["high_conf"] = run("pitcher trackman (high_conf mapping only)", high_conf, fit_df, holdout_df, trackman_path)
    results["all_resolved"] = run("pitcher trackman (high_conf + single-season mapping)", final_map, fit_df, holdout_df, trackman_path)

    print("\n\n=== SUMMARY (cross-regime holdout, 2019-2023 -> 2024) ===")
    print(f"{'variant':15s} {'holdout(raw)':>14s} {'holdout(cal)':>14s}")
    for k, (raw, cal) in results.items():
        print(f"{k:15s} {raw:14.1f} {cal:14.1f}")


if __name__ == "__main__":
    main()
