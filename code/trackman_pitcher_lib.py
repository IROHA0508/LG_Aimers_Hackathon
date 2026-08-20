# trackman_pitcher_lib.py
# ------------------------------------------------------------
# trackman_pitcher_matching.py에서 만든 워크로드+손잡이 기반 Hungarian 매칭
# 로직을 임의의 cutoff_season에 대해 재사용 가능하게 함수화한 것.
# walk-forward의 각 폴드는 fit 마지막 시즌이 다르므로(2019/2019-20/.../2019-23),
# 매 폴드마다 그 시점까지의 데이터로 다시 매칭해야 시점 리크가 없다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import Counter

TEAM_MAP = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA", 16: "KIA_TIG",
            17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN", 20: "KT_WIZ", 21: "SK_WYV"}


def _match_team_season(train, tm, team_id, season):
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


def build_pitcher_id_map(train, tm, min_season, cutoff_season, max_ratio=0.15):
    """min_season..cutoff_season 구간의 모든 (team,season)에서 매칭 후 시즌간
    일관성으로 신뢰도를 매겨 최종 pitcher_id -> pitcher_trackman_id 매핑을 만든다.
    (high_conf: 2개 이상 시즌 100% 일치 / single: 1개 시즌만 관측, ratio<max_ratio)
    둘 다 합친 "all_resolved"가 trackman_pitcher_feature_check.py에서 가장 나았음."""
    all_matches = []
    for team_id in TEAM_MAP:
        for season in range(min_season, cutoff_season + 1):
            m = _match_team_season(train, tm, team_id, season)
            if len(m):
                all_matches.append(m)
    if not all_matches:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id"])
    matches = pd.concat(all_matches, ignore_index=True)
    matches["ratio"] = matches["cost"] / matches["n_train"]
    good = matches[matches["ratio"] < max_ratio].copy()

    rows = []
    for pid, grp in good.groupby("pitcher_id"):
        counts = Counter(grp["pitcher_trackman_id"])
        best_tid, n_agree = counts.most_common(1)[0]
        n_seasons = grp["season"].nunique()
        rows.append(dict(pitcher_id=pid, pitcher_trackman_id=best_tid,
                          n_seasons_seen=n_seasons, agree_rate=n_agree / n_seasons))
    resolved = pd.DataFrame(rows)
    if len(resolved) == 0:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id"])
    return resolved[["pitcher_id", "pitcher_trackman_id"]]


def build_trackman_pitcher_prior(tm, cutoff_season, id_map_df):
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

    prior = id_map_df.merge(prior, on="pitcher_trackman_id", how="left")
    prior = prior.drop(columns=["pitcher_trackman_id"])
    return prior


def build_trackman_consistency_prior(tm, cutoff_season, id_map_df):
    """기존 build_trackman_pitcher_prior는 구속/회전/무브먼트/구종비율의 "평균"(스터프
    자체)을 썼는데, 이는 팀/개인 레벨 모두 walk-forward에서 손해로 판명됐다
    (trackman에 로케이션/결과값이 없어 asof_pitcher_* 성공률 이력보다 타겟과 간접적으로만
    관련되기 때문으로 추정). 이 함수는 대신 "제구 일관성"에 더 가까운 대리 신호를 만든다:
    - 릴리스포인트(rel_height/rel_side) 표준편차: 폼이 일정할수록 낮음 -> 커맨드 대리 지표
    - 익스텐션 표준편차: 릴리스 타이밍 일관성
    - 패스트볼 그룹 내 구속/회전/무브먼트 표준편차: 가장 많이 던지는 구종 하나로 좁혀
      노이즈를 줄이면서 "같은 공을 얼마나 똑같이 던지는가"를 포착
    평균(스터프) 대신 분산(일관성)이라 asof_pitcher_success_rate류와 정보가 덜 겹칠 것으로
    기대되지만, 매칭 커버리지(개인 단위 ~37%) 문제는 그대로 남아있어 walk-forward로
    반드시 재검증해야 한다."""
    tm = tm[tm["season"] <= cutoff_season]

    overall = tm.groupby("pitcher_trackman_id").agg(
        tm_n=("pitch_type_group", "size"),
        tm_relheight_std=("rel_height", "std"),
        tm_relside_std=("rel_side", "std"),
        tm_extension_std=("extension", "std"),
    ).reset_index()

    fb = tm[tm["pitch_type_group"] == "fastball"]
    fb_agg = fb.groupby("pitcher_trackman_id").agg(
        tm_fb_n=("rel_speed", "size"),
        tm_fb_relspeed_std=("rel_speed", "std"),
        tm_fb_ivb_std=("induced_vert_break", "std"),
        tm_fb_hb_std=("horz_break", "std"),
        tm_fb_spin_std=("spin_rate", "std"),
    ).reset_index()

    prior = overall.merge(fb_agg, on="pitcher_trackman_id", how="left")
    prior = id_map_df.merge(prior, on="pitcher_trackman_id", how="left")
    prior = prior.drop(columns=["pitcher_trackman_id"])
    return prior


def attach_trackman_consistency_features(fit_df, holdout_df, tm, cutoff_season):
    """attach_trackman_pitcher_features와 동일한 매칭 인프라(build_pitcher_id_map)를
    쓰되, prior 테이블만 build_trackman_consistency_prior로 교체한 버전."""
    id_map = build_pitcher_id_map(fit_df, tm, min_season=fit_df["season"].min(), cutoff_season=cutoff_season)
    prior = build_trackman_consistency_prior(tm, cutoff_season, id_map)
    fit_out = fit_df.merge(prior, on="pitcher_id", how="left")
    holdout_out = holdout_df.merge(prior, on="pitcher_id", how="left")
    match_rate_fit = fit_out["tm_n"].notna().mean() if "tm_n" in fit_out else 0.0
    match_rate_holdout = holdout_out["tm_n"].notna().mean() if "tm_n" in holdout_out else 0.0
    return fit_out, holdout_out, dict(n_mapped=len(id_map), match_rate_fit=match_rate_fit,
                                       match_rate_holdout=match_rate_holdout)


def attach_trackman_pitcher_features(fit_df, holdout_df, tm, cutoff_season):
    """fit_df/holdout_df에 trackman 개인 피처를 붙인다. cutoff_season은 fit 기간의
    마지막 시즌(=holdout 시즌-1)이어야 시점 리크가 없다. tm['team_merged']는
    미리 SSG_LAN->SK_WYV 치환이 돼 있어야 함."""
    id_map = build_pitcher_id_map(fit_df, tm, min_season=fit_df["season"].min(), cutoff_season=cutoff_season)
    prior = build_trackman_pitcher_prior(tm, cutoff_season, id_map)
    fit_out = fit_df.merge(prior, on="pitcher_id", how="left")
    holdout_out = holdout_df.merge(prior, on="pitcher_id", how="left")
    match_rate_fit = fit_out["tm_n"].notna().mean() if "tm_n" in fit_out else 0.0
    match_rate_holdout = holdout_out["tm_n"].notna().mean() if "tm_n" in holdout_out else 0.0
    return fit_out, holdout_out, dict(n_mapped=len(id_map), match_rate_fit=match_rate_fit,
                                       match_rate_holdout=match_rate_holdout)
