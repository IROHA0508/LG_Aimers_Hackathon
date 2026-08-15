# LG_Aimers_Hackathon
LG Aimers 9기 해커톤 프로젝트 (DACON #236743, KBO 투구 제구 성공 확률 예측)

## 평가지표
`Score = max(0, 100000 * (1 - Brier / baseline))`, baseline = r*(1-r), r = mean(control_success) ≈ 0.5238
전체 데이터(147만행) 기준 baseline_brier ≈ 0.249446. 판별력보다 "확률 보정 품질"에 민감한 지표라
Isotonic 보정을 파이프라인 필수 단계로 둠.

## 피처 엔지니어링 결론 (`code/train_ensemble.py`의 `build_features`)
전체 데이터(147만행) 기준 LightGBM Brier로 채택/기각 판단. as-of(시점 리크 방지) 방식 공통 적용.

| 피처 | 결과 | 채택 |
|---|---|---|
| 팀 단위 as-of 성공률 (`use_team_feature`) | 0.24405 → 0.24382 (+0.10%) | ✅ |
| 팀x팀 맞대결 + 시즌 추세보정 (`use_team_matchup_feature`, `use_season_trend_feature`) | 0.243817 → 0.24376 (+0.02%) | ✅ |
| 투수-타자 개인 맞대결 (`use_matchup_feature`) | -0.07% (악화) | ❌ 표본 희소(조합당 평균 15행) |
| 압박상황(풀카운트) 클러치 성향 (`use_clutch_feature`) | -0.01% (악화) | ❌ 표본 희소(투수당 중앙값 47행) |
| 트랙맨 이력 (`USE_TRACKMAN`) | pitcher_id 매칭률 0% | ❌ ID 네임스페이스 불일치로 연결 불가 |

최종 82개 피처 사용.

## 모델 하이퍼파라미터 튜닝 (2단계 순차 그리드서치: 2-fold 탐색 → 5-fold 최종검증)

| 모델 | 기본값 Brier | 튜닝 Brier | 개선(점수) |
|---|---|---|---|
| LightGBM | 0.243759 | 0.243454 (3라운드) | +122.2 |
| XGBoost | 0.243809 | 0.243708 (1라운드) | +40.5 |
| CatBoost | - | - | 튜닝 스킵 (학습 시간 대비 수확체감 판단) |

LGB는 라운드마다 개선폭이 줄어들어(+54.3 → +50.8 → +17.1) 3라운드에서 중단. CatBoost는 기본값 학습만도
느려서 튜닝을 진행하지 않음.

## 앙상블 구성 실험 (전체 데이터, 5-fold 기준, 2026-08-15~16)

| 조합 | Score | 비고 |
|---|---|---|
| LightGBM 단독 | 2397.8 | 튜닝된 기본 성능 |
| XGBoost 단독 | 2295.9 | |
| CatBoost 단독 | 2232.9 | |
| RandomForest 단독 | 2071.4 | |
| Entity-Embedding MLP 단독 | 1841.9 | |
| LogisticRegression 단독 | 1393.7 | |
| LGB+XGB+CAT 3way 스태킹 | 2381.9 | LGB 단독보다 **-15.9** (희석효과) |
| LGB+RF 2way 스태킹 | 2409.2 | LGB 단독보다 **+11.4** |
| LGB+RF+LR 3way 스태킹 | 2394.6 | LGB+RF 대비 **-14.5** (LR: 다양성도 크지 않은데 성능만 나빠 희석) |
| **LGB+MLP 2way 스태킹** | **2411.4** | LGB 단독보다 **+13.7**, 실험 중 최고점 → **최종 채택** |
| LGB+RF+MLP 3way 스태킹 | 2411.6 | LGB+MLP 대비 **+0.1** (RF-MLP 상관계수 0.901로 중복 → 기각) |

### 핵심 교훈
스태킹 이득은 "각 모델의 단독 성능"이 아니라 **"LGB와의 OOF 예측 상관관계"**가 결정한다.
- XGB/CAT: LGB와 같은 그래디언트 부스팅 계열이라 상관관계가 높음(추정) → 스태킹 시 손해
- RF: 배깅 기반, lgb-rf 상관계수 0.939 → 단독 성능 최하위권이어도 소폭 이득(+11.4)
- MLP: 신경망 기반, lgb-mlp 상관계수 0.866(RF보다 낮음) → RF보다 단독 성능은 낮지만 더 큰 이득(+13.7)
- LR: lgb-lr 상관계수 0.759로 가장 낮지만, 단독 성능이 너무 낮아(1393.7) 다양성 이득을 다 까먹음 → 손해
- RF+MLP 동시 사용: rf-mlp 상관계수 0.901로 서로 중복 → 3way 추가 이득 없음

**결론: "다양성(낮은 상관관계)이 있으면서 동시에 최소한의 단독 성능을 갖춘" 모델만 스태킹에 도움이 된다.**
너무 강한 모델(XGB/CAT, LGB와 유사 계열)이나 너무 약한 모델(LR)은 둘 다 손해를 봄.

## 최종 아키텍처
**LightGBM(튜닝됨) + Entity-Embedding MLP 2모델 스태킹** (`code/train_ensemble.py`)
- LGB: `train_lgb` (2단계 순차 그리드서치로 튜닝된 하이퍼파라미터)
- MLP: `train_mlp` (범주형 임베딩 + 수치형 표준화 → MLP, torch/GPU)
- Isotonic 보정 → Logistic 메타러너 스태킹 → 스태킹 결과 재보정
- 최종 검증 점수: **2411.4** (baseline LightGBM 기본값 대비 세션 전체 개선폭 대부분 반영)

`train_xgb`/`train_cat`/`train_rf`/`train_lr` 함수는 실험 근거 보존을 위해 코드에 남겨두었으나
메인 파이프라인(`main()`)에서는 사용하지 않음.

### 실험 스크립트
- `code/grid_search_lgb*.py`, `code/grid_search_xgb.py`: 하이퍼파라미터 그리드서치
- `code/final_check.py`: LGB+XGB+CAT 3way 검증 (초기 아키텍처, 기각)
- `code/rf_check.py`, `code/lr_check.py`, `code/mlp_check.py`, `code/full_stack_check.py`: 앙상블 조합 검증
