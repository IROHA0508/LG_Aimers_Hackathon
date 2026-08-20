# walkforward_stacking_check3.py
# ------------------------------------------------------------
# session1은 랜덤fold 기준 "LGB+XGB+CAT 3way가 LGB 단독보다 -15.9 손해"
# (같은 그래디언트 부스팅 계열이라 상관관계가 높아서라고 추정)라고 판단해
# 기각했다. 이걸 walk-forward 기준으로 재검증한다: XGB/CAT 단독 성능,
# LGB와의 OOF 상관관계, 그리고 LGB+XGB/LGB+CAT/LGB+XGB+CAT 조합.
#
# 지금까지 확정된 최선 조합(LGB+RF+MLP 3way, walk-forward 평균 934.0)에
# XGB/CAT을 추가하는 게 의미있는지도 마지막에 5-way로 확인한다.
# 피처 설정은 이번 세션 확정안(team/team_matchup/season_trend OFF) 유지.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import train_ensemble as te
from walkforward_harness import FOLD_SEASONS, score, DATA_DIR
from walkforward_stacking_check import prep_fold
from walkforward_stacking_check2 import fit_stack

TARGET_COL = te.TARGET_COL

LGB_ALONE_SCORES = {2020: 286.6, 2021: 1422.3, 2022: 2351.3, 2023: 0.0, 2024: 457.8}
LGB_RF_MLP_SCORES = {2020: 359.2, 2021: 1447.9, 2022: 2382.7, 2023: 0.0, 2024: 480.3}


def xgb_predict(model, X):
    d = xgb.DMatrix(X, enable_categorical=True)
    return model.predict(d, iteration_range=(0, model.best_iteration + 1))


def run_fold(train, target_season):
    X_fit, y_fit, X_holdout, y_holdout, cat_features = prep_fold(train, target_season)

    lgb_models, lgb_oof = te.train_lgb(X_fit, y_fit, X_fit, cat_features, verbose_fold=False)
    lgb_pred_h = np.mean([m.predict(X_holdout, num_iteration=m.best_iteration) for m in lgb_models], axis=0)

    xgb_models, xgb_oof = te.train_xgb(X_fit, y_fit, cat_features, verbose_fold=False)
    xgb_pred_h = np.mean([xgb_predict(m, X_holdout) for m in xgb_models], axis=0)

    cat_models, cat_oof = te.train_cat(X_fit, y_fit, cat_features)
    cat_pred_h = np.mean([m.predict_proba(X_holdout)[:, 1] for m in cat_models], axis=0)

    def alone_score(oof, pred_h):
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_fit)
        return score(brier_score_loss(y_holdout, iso.predict(pred_h)))

    lgb_xgb_corr = np.corrcoef(lgb_oof, xgb_oof)[0, 1]
    lgb_cat_corr = np.corrcoef(lgb_oof, cat_oof)[0, 1]

    xgb_only = alone_score(xgb_oof, xgb_pred_h)
    cat_only = alone_score(cat_oof, cat_pred_h)

    lgb_xgb = score(brier_score_loss(y_holdout, fit_stack(
        {"lgb": lgb_oof, "xgb": xgb_oof}, y_fit, {"lgb": lgb_pred_h, "xgb": xgb_pred_h})))
    lgb_cat = score(brier_score_loss(y_holdout, fit_stack(
        {"lgb": lgb_oof, "cat": cat_oof}, y_fit, {"lgb": lgb_pred_h, "cat": cat_pred_h})))
    lgb_xgb_cat = score(brier_score_loss(y_holdout, fit_stack(
        {"lgb": lgb_oof, "xgb": xgb_oof, "cat": cat_oof}, y_fit,
        {"lgb": lgb_pred_h, "xgb": xgb_pred_h, "cat": cat_pred_h})))

    return dict(target_season=target_season, lgb_xgb_corr=lgb_xgb_corr, lgb_cat_corr=lgb_cat_corr,
                xgb_only=xgb_only, cat_only=cat_only, lgb_xgb=lgb_xgb, lgb_cat=lgb_cat,
                lgb_xgb_cat=lgb_xgb_cat)


def main():
    print("Load train...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")

    results = []
    for target_season in FOLD_SEASONS:
        r = run_fold(train, target_season)
        lgb_ref = LGB_ALONE_SCORES[target_season]
        print(f"predict {target_season}: corr(lgb,xgb)={r['lgb_xgb_corr']:.3f} corr(lgb,cat)={r['lgb_cat_corr']:.3f} | "
              f"LGB={lgb_ref:.1f}  XGB={r['xgb_only']:.1f}  CAT={r['cat_only']:.1f}  "
              f"LGB+XGB={r['lgb_xgb']:.1f}  LGB+CAT={r['lgb_cat']:.1f}  LGB+XGB+CAT={r['lgb_xgb_cat']:.1f}")
        results.append(r)

    lgb_scores = np.array([LGB_ALONE_SCORES[r["target_season"]] for r in results])
    xgb_scores = np.array([r["xgb_only"] for r in results])
    cat_scores = np.array([r["cat_only"] for r in results])
    lgbxgb_scores = np.array([r["lgb_xgb"] for r in results])
    lgbcat_scores = np.array([r["lgb_cat"] for r in results])
    lgbxgbcat_scores = np.array([r["lgb_xgb_cat"] for r in results])

    print("\n=== SUMMARY (5-fold mean) ===")
    print(f"LGB alone:         {lgb_scores.mean():.1f}")
    print(f"XGB alone:         {xgb_scores.mean():.1f}")
    print(f"CAT alone:         {cat_scores.mean():.1f}")
    print(f"LGB+XGB:           {lgbxgb_scores.mean():.1f}")
    print(f"LGB+CAT:           {lgbcat_scores.mean():.1f}")
    print(f"LGB+XGB+CAT 3way:  {lgbxgbcat_scores.mean():.1f}")
    print(f"(reference) LGB+RF+MLP 3way: {np.mean(list(LGB_RF_MLP_SCORES.values())):.1f}")


if __name__ == "__main__":
    main()
