# trackman_pitcher_matching.py
# ------------------------------------------------------------
# 트랙맨 팀 레벨 피처는 정보량 부족으로 실패했음(README 세션2). 남은 마지막
# 후보: 선수 개인 단위 연결. 팀×시즌 그룹 내에서 투구량(workload) + 손잡이를
# 코스트로 한 Hungarian 최적 매칭(scipy.optimize.linear_sum_assignment)으로
# pitcher_id <-> pitcher_trackman_id를 복원한다.
#
# 절차:
#  1) fit 기간(2019~2023) 전체의 (team_id, season) 조합마다 워크로드+손잡이
#     기반 최적 매칭을 수행해 매칭 후보를 모은다.
#  2) 한 선수가 여러 시즌에 걸쳐 나타나면(이적 없는 한 팀에 소속), 시즌마다
#     독립적으로 매칭한 pitcher_trackman_id가 얼마나 일관되는지로 신뢰도를
#     판단한다 - 같은 실제 선수라면 매 시즌 같은 trackman_id로 매칭돼야 함.
#  3) 신뢰도 높은 매칭만 채택해 pitcher_id -> pitcher_trackman_id 최종
#     테이블을 만든다.
#  4) 검증(매칭에 전혀 안 쓰인 독립 신호): train.csv 자체가 제공하는
#     asof_pitcher_fastball_rate/breaking_rate/offspeed_rate와 매칭된
#     trackman 투수의 실제 구종 비율(tm_mix_*)이 상관관계가 있는지 확인.
#     매칭이 맞다면 강한 양의 상관관계가 나와야 한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import Counter

DATA_DIR = "../open/data"

TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA", 16: "KIA_TIG",
            17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN", 20: "KT_WIZ", 21: "SK_WYV"}


def match_team_season(train, tm, team_id, season):
    code = TEAM_MAP[team_id]
    t_sub = train[(train.pitcher_team_id == team_id) & (train.season == season)]
    tc = t_sub.groupby("pitcher_id").agg(n=("row_id", "size"), hand=("pitcher_hand", "first")).reset_index()
    m_sub = tm[(tm.team_merged == code) & (tm.season == season)]
    mc = m_sub.groupby("pitcher_trackman_id").agg(n=("trackman_id", "size"), hand=("pitcher_hand", "first")).reset_index()
    if len(tc) == 0 or len(mc) == 0:
        return pd.DataFrame()

    scale = tc["n"].sum() / mc["n"].sum()
    mc["n_scaled"] = mc["n"] * scale
    tc_n = tc["n"].values
    tc_hand = (tc["hand"].values == 1)
    mc_n = mc["n_scaled"].values
    mc_hand = (mc["hand"].values == "Left")

    diff = np.abs(tc_n[:, None] - mc_n[None, :])
    mismatch = tc_hand[:, None] != mc_hand[None, :]
    cost = diff + mismatch * 1e6

    r_ind, c_ind = linear_sum_assignment(cost)
    out = []
    for r, c in zip(r_ind, c_ind):
        out.append(dict(pitcher_id=tc.iloc[r]["pitcher_id"], pitcher_trackman_id=mc.iloc[c]["pitcher_trackman_id"],
                         season=season, team_id=team_id, cost=cost[r, c], n_train=tc.iloc[r]["n"]))
    return pd.DataFrame(out)


def build_matches(train, tm, cutoff_season):
    all_matches = []
    for team_id in TEAM_MAP:
        for season in range(2019, cutoff_season + 1):
            m = match_team_season(train, tm, team_id, season)
            if len(m):
                all_matches.append(m)
    matches = pd.concat(all_matches, ignore_index=True)
    matches["ratio"] = matches["cost"] / matches["n_train"]
    return matches


def resolve_consensus(matches, max_ratio=0.15):
    good = matches[matches["ratio"] < max_ratio].copy()
    rows = []
    for pid, grp in good.groupby("pitcher_id"):
        counts = Counter(grp["pitcher_trackman_id"])
        best_tid, n_agree = counts.most_common(1)[0]
        n_seasons = grp["season"].nunique()
        total_n_train = grp.loc[grp["pitcher_trackman_id"] == best_tid, "n_train"].sum()
        rows.append(dict(pitcher_id=pid, pitcher_trackman_id=best_tid,
                          n_seasons_seen=n_seasons, n_seasons_agree=n_agree,
                          agree_rate=n_agree / n_seasons, total_n_train=total_n_train))
    resolved = pd.DataFrame(rows)
    return resolved


def main():
    print("Load data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    tm["team_merged"] = tm["pitcher_team"].replace({"SSG_LAN": "SK_WYV"})

    CUTOFF = 2023  # fit 기간 마지막 시즌 (cross-regime holdout과 동일 기준)
    print(f"Matching pitchers for seasons 2019-{CUTOFF}...")
    matches = build_matches(train, tm, CUTOFF)
    print(f"raw matches: {len(matches)}  (low-cost ratio<0.15: {(matches['ratio']<0.15).sum()})")

    resolved = resolve_consensus(matches, max_ratio=0.15)
    print(f"\nresolved unique pitcher_id -> pitcher_trackman_id mappings: {len(resolved)}")
    print(resolved["agree_rate"].value_counts().sort_index())

    # 고신뢰: 2개 이상 시즌에서 나타났고, 매 시즌 같은 trackman_id로 일치(agree_rate==1.0)
    high_conf = resolved[(resolved["n_seasons_seen"] >= 2) & (resolved["agree_rate"] == 1.0)]
    # 저신뢰(1개 시즌만 관측): 그 자체 매칭 비용이 낮으면 채택
    single_season = resolved[resolved["n_seasons_seen"] == 1]
    print(f"\nhigh_conf (>=2 seasons, 100% agree): {len(high_conf)}")
    print(f"single_season only: {len(single_season)}")

    final_map = pd.concat([high_conf, single_season], ignore_index=True)
    print(f"\nfinal accepted mappings: {len(final_map)} (train.csv 전체 고유 pitcher_id={train['pitcher_id'].nunique()})")

    final_map.to_csv("trackman_pitcher_map.csv", index=False)
    print("saved -> code/trackman_pitcher_map.csv")

    # ---- 독립 검증: train 자체의 구종 비율 asof 피처 vs 매칭된 trackman 실제 구종 비율 ----
    print("\n=== Validation: matched trackman pitch-mix vs train's own asof pitch-mix ===")
    train_pmix = train.groupby("pitcher_id").agg(
        fastball=("asof_pitcher_fastball_rate", "last"),
        breaking=("asof_pitcher_breaking_rate", "last"),
        offspeed=("asof_pitcher_offspeed_rate", "last"),
        n=("asof_pitcher_pitchmix_n", "last"),
    ).reset_index()

    tm_pmix = (tm[tm.season <= CUTOFF].groupby(["pitcher_trackman_id", "pitch_type_group"])
               .size().unstack(fill_value=0))
    tm_pmix = tm_pmix.div(tm_pmix.sum(axis=1), axis=0)
    tm_pmix = tm_pmix.rename(columns={"fastball": "tm_fastball", "breaking": "tm_breaking", "offspeed": "tm_offspeed"})
    tm_pmix = tm_pmix.reset_index()

    check = final_map.merge(train_pmix, on="pitcher_id", how="left").merge(tm_pmix, on="pitcher_trackman_id", how="left")
    check = check.dropna(subset=["fastball", "tm_fastball"])
    check = check[check["n"] >= 30]  # 표본 너무 적은 건 노이즈라 제외
    print(f"validation sample size (n>=30 pitches): {len(check)}")
    print(f"corr(fastball_rate): {check['fastball'].corr(check['tm_fastball']):.3f}")
    print(f"corr(breaking_rate): {check['breaking'].corr(check['tm_breaking']):.3f}")
    if "offspeed" in tm_pmix.columns:
        print(f"corr(offspeed_rate): {check['offspeed'].corr(check['tm_offspeed']):.3f}")


if __name__ == "__main__":
    main()
