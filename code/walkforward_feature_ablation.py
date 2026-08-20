# walkforward_feature_ablation.py
# ------------------------------------------------------------
# 세션1에서 랜덤 K-fold로 결정된 피처 채택/기각(README):
#   team_feature: 채택(+0.10%), team_matchup+season_trend: 채택(+0.02%),
#   individual matchup: 기각(-0.07%), clutch: 기각(-0.01%)
# 이걸 5-fold 확장 윈도우(walk-forward)로 재검증한다. 계산량 관리를 위해
# 스크리닝 단계는 내부 K-fold를 5->2로 낮춘다(세션1도 원래 "2-fold 탐색 ->
# 5-fold 최종검증" 2단계 방식을 썼음 - 같은 논리).
#
# baseline도 2-fold로 다시 돌려서 공정 비교(walkforward_harness.py의 5-fold
# baseline과는 내부 fold 수가 달라 절대값이 다를 수 있음 - 상대 비교만 유효).
# ------------------------------------------------------------
import pandas as pd

from walkforward_harness import walk_forward_eval, DATA_DIR

CONFIGS = {
    "baseline (all default)": {},
    "team_feature OFF": dict(use_team_feature=False),
    "team_matchup_feature OFF": dict(use_team_matchup_feature=False),
    "season_trend_feature OFF": dict(use_season_trend_feature=False),
    "individual matchup ON": dict(use_matchup_feature=True),
    "clutch ON": dict(use_clutch_feature=True),
}


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    all_results = {}
    for name, kwargs in CONFIGS.items():
        rows = walk_forward_eval(train, feature_kwargs=kwargs, n_internal_folds=2, name=name)
        all_results[name] = rows

    print("\n\n################ FINAL SUMMARY (2-fold internal screening) ################")
    print(f"{'config':30s} " + " ".join(f"{s:>8d}" for s in [2020, 2021, 2022, 2023, 2024]) + f" {'mean':>10s}")
    for name, rows in all_results.items():
        vals = [r["holdout_score_cal"] for r in rows]
        mean = sum(vals) / len(vals)
        print(f"{name:30s} " + " ".join(f"{v:8.1f}" for v in vals) + f" {mean:10.1f}")


if __name__ == "__main__":
    main()
