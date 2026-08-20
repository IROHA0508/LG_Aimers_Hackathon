# walkforward_stacking_check2.py
# ------------------------------------------------------------
# walkforward_stacking_check.py에서 LGB+MLP 2way 스태킹이 walk-forward
# 기준으로도 진짜 효과(+14.5, 903.6->918.1)임을 확인했다. 이번엔 세션1이
# 검증했던 다른 조합(LGB+RF 2way, LGB+RF+MLP 3way)을 같은 walk-forward
# 5-fold 기준으로 재검증한다. RF는 session1 랜덤fold 기준 lgb-rf 상관계수
# 0.939(MLP의 0.866보다 높음)로 단독으로는 MLP보다 스태킹 이득이 작을 것으로
# 예상되지만, forward 기준으로도 그런지 직접 확인한다.
#
# 피처 설정은 이번 세션에서 확정한 개선안(team/team_matchup/season_trend
# 피처 OFF) 유지. LGB/MLP 결과는 walkforward_stacking_check.py에서 이미
# 확보(재사용), RF만 새로 학습한다.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR
from walkforward_stacking_check import prep_fold, FEATURE_KWARGS

TARGET_COL = te.TARGET_COL

LGB_ALONE_SCORES = {2020: 286.6, 2021: 1422.3, 2022: 2351.3, 2023: 0.0, 2024: 457.8}
LGB_MLP_STACK_SCORES = {2020: 325.7, 2021: 1415.9, 2022: 2365.4, 2023: 0.0, 2024: 483.4}


def fit_stack(oof_dict, y_fit, pred_dict):
    """oof_dict/pred_dict: {model_name: array}. 각자 isotonic 보정 후 LogisticRegression
    메타러너로 스태킹, 스태킹 결과를 다시 isotonic 보정."""
    calibrators = {}
    calibrated_oof = {}
    for name, oof in oof_dict.items():
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, y_fit)
        calibrators[name] = iso
        calibrated_oof[name] = iso.predict(oof)

    names = list(oof_dict.keys())
    meta_X = np.column_stack([calibrated_oof[n] for n in names])
    meta = LogisticRegression()
    meta.fit(meta_X, y_fit)
    stack_pred_fit = meta.predict_proba(meta_X)[:, 1]

    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(stack_pred_fit, y_fit)

    calibrated_pred = {n: calibrators[n].predict(pred_dict[n]) for n in names}
    meta_X_h = np.column_stack([calibrated_pred[n] for n in names])
    stack_pred_h = meta.predict_proba(meta_X_h)[:, 1]
    final_pred_h = final_iso.predict(stack_pred_h)
    return final_pred_h


def run_fold(train, target_season):
    X_fit, y_fit, X_holdout, y_holdout, cat_features = prep_fold(train, target_season)

    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    lgb_pred_h = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)

    mlp_state_dicts, mlp_oof, mlp_preproc = te.train_mlp(X_fit, y_fit, cat_features, verbose_fold=False)
    mlp_pred_h = te.predict_mlp(mlp_state_dicts, mlp_preproc, X_holdout)

    rf_models, rf_oof = te.train_rf(X_fit, y_fit, cat_features)
    rf_pred_h = np.mean([pipe.predict_proba(X_holdout)[:, 1] for pipe in rf_models], axis=0)

    lgb_rf_corr = np.corrcoef(lgb_oof, rf_oof)[0, 1]
    rf_iso = IsotonicRegression(out_of_bounds="clip").fit(rf_oof, y_fit)
    rf_only_score = score(brier_score_loss(y_holdout, rf_iso.predict(rf_pred_h)))

    lgb_rf_pred = fit_stack({"lgb": lgb_oof, "rf": rf_oof}, y_fit, {"lgb": lgb_pred_h, "rf": rf_pred_h})
    lgb_rf_score = score(brier_score_loss(y_holdout, lgb_rf_pred))

    lgb_rf_mlp_pred = fit_stack({"lgb": lgb_oof, "rf": rf_oof, "mlp": mlp_oof}, y_fit,
                                 {"lgb": lgb_pred_h, "rf": rf_pred_h, "mlp": mlp_pred_h})
    lgb_rf_mlp_score = score(brier_score_loss(y_holdout, lgb_rf_mlp_pred))

    return dict(target_season=target_season, lgb_rf_corr=lgb_rf_corr, rf_only=rf_only_score,
                lgb_rf=lgb_rf_score, lgb_rf_mlp=lgb_rf_mlp_score)


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    results = []
    for target_season in FOLD_SEASONS:
        r = run_fold(train, target_season)
        lgb_ref = LGB_ALONE_SCORES[target_season]
        lgbmlp_ref = LGB_MLP_STACK_SCORES[target_season]
        print(f"predict {target_season}: lgb_rf_corr={r['lgb_rf_corr']:.3f} | "
              f"LGB={lgb_ref:.1f}  RF={r['rf_only']:.1f}  LGB+RF={r['lgb_rf']:.1f}  "
              f"LGB+MLP(ref)={lgbmlp_ref:.1f}  LGB+RF+MLP={r['lgb_rf_mlp']:.1f}")
        results.append(r)

    lgb_scores = np.array([LGB_ALONE_SCORES[r["target_season"]] for r in results])
    lgbmlp_scores = np.array([LGB_MLP_STACK_SCORES[r["target_season"]] for r in results])
    rf_scores = np.array([r["rf_only"] for r in results])
    lgbrf_scores = np.array([r["lgb_rf"] for r in results])
    lgbrfmlp_scores = np.array([r["lgb_rf_mlp"] for r in results])

    print("\n=== SUMMARY (5-fold mean) ===")
    print(f"LGB alone:        {lgb_scores.mean():.1f}")
    print(f"RF alone:         {rf_scores.mean():.1f}")
    print(f"LGB+MLP (ref):    {lgbmlp_scores.mean():.1f}")
    print(f"LGB+RF:           {lgbrf_scores.mean():.1f}  (delta vs LGB alone: {lgbrf_scores.mean()-lgb_scores.mean():+.1f})")
    print(f"LGB+RF+MLP 3way:  {lgbrfmlp_scores.mean():.1f}  (delta vs LGB+MLP: {lgbrfmlp_scores.mean()-lgbmlp_scores.mean():+.1f})")


if __name__ == "__main__":
    main()
