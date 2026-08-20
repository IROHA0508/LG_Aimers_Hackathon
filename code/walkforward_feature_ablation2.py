# walkforward_feature_ablation2.py
# ------------------------------------------------------------
# walkforward_feature_ablation.py에서 team_feature/team_matchup_feature/
# season_trend_feature를 각각 끄면 baseline보다 forward 성능이 나은 것을
# 확인했다(2-fold 스크리닝, 평균 741.1 -> 801.7/763.3/773.1). 세 개를
# 동시에 끄면 더 나은지 확인하고, 유의미하면 5-fold 내부검증으로 최종
# 확인한다.
# ------------------------------------------------------------
import pandas as pd

from walkforward_harness import walk_forward_eval, DATA_DIR

SCREEN_CONFIGS = {
    "all three OFF (team+team_matchup+season_trend)": dict(
        use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False),
}


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    print("\n### Stage 1: 2-fold screening ###")
    screen_results = {}
    for name, kwargs in SCREEN_CONFIGS.items():
        rows = walk_forward_eval(train, feature_kwargs=kwargs, n_internal_folds=2, name=name)
        screen_results[name] = rows

    print("\n\n### Stage 2: 5-fold confirmation (baseline vs all-three-off) ###")
    confirm_results = {}
    confirm_results["baseline (5-fold)"] = walk_forward_eval(
        train, feature_kwargs={}, n_internal_folds=5, name="baseline (5-fold confirm)")
    confirm_results["all three OFF (5-fold)"] = walk_forward_eval(
        train, feature_kwargs=dict(use_team_feature=False, use_team_matchup_feature=False,
                                    use_season_trend_feature=False),
        n_internal_folds=5, name="all three OFF (5-fold confirm)")

    print("\n\n################ FINAL SUMMARY ################")
    print(f"{'config':40s} " + " ".join(f"{s:>8d}" for s in [2020, 2021, 2022, 2023, 2024]) + f" {'mean':>10s}")
    for name, rows in {**screen_results, **confirm_results}.items():
        vals = [r["holdout_score_cal"] for r in rows]
        mean = sum(vals) / len(vals)
        print(f"{name:40s} " + " ".join(f"{v:8.1f}" for v in vals) + f" {mean:10.1f}")


if __name__ == "__main__":
    main()
