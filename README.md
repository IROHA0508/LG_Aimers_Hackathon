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

## 세션 2 (2026-08-16): 검증 방법론 재점검

### 발단
`submit.zip` 실제 리더보드 제출 결과 **786점** — 로컬에서 report한 CV 점수(2411.4)와 3배 이상 차이.
원인 진단과 개선을 위해 아래 실험들을 순서대로 진행. 실험 스크립트: `code/time_holdout_check.py`,
`code/time_holdout_recalibration.py`, `code/regime_hypothesis_check.py`, `code/regularization_search.py`,
`code/regularization_confirm_crossregime.py`, `code/trackman_team_feature_check.py`.

### 핵심 진단: CV-LB 격차의 원인은 검증 방법론 자체
`test.csv`(실제 평가 데이터)는 `season=2025`로 train.csv(2019~2024)에 전혀 없던 미래 시즌인데,
기존 검증은 `StratifiedKFold(shuffle=True)`로 2019~2024를 무작위로 섞어 평가 — "미래 시즌 외삽"을
전혀 검증하지 못하는 방식이었음. **시간 기반 홀드아웃**(2019-2023 학습 → 2024를 미래처럼 검증,
LGB 단독 기준)으로 재현한 결과:

| 구분 | Brier | Score |
|---|---|---|
| fit(2019-2023) 내부 랜덤 5-fold (보정 후) | 0.24281 | 2659.5 |
| **2024 홀드아웃**(한 번도 안 본 시즌, fit 보정기 적용) | 0.24830 | **460.2** |

랜덤 fold 대비 5.8배 붕괴 — 실제 제출 격차(2411.4→786, 3.1배)와 같은 방향·같은 규모.
**이후 모든 실험은 이 cross-regime 홀드아웃(2019-2023→2024)을 기준으로 판단.**
이 지표는 baseline_brier(0.249446)에 아주 가까운 영역에서 작동해 미세한 Brier 차이도
score를 크게 흔든다는 점에 유의(예: brier 0.003 차이가 score 1000+ 차이로 증폭됨).

### 시도했지만 효과 없었던 것들

| 시도 | 방법 | 결과 (holdout score, 보정 후) | 결론 |
|---|---|---|---|
| Prior-shift 재보정 | 시즌 추세로 외삽한 목표 성공률로 Bayes 보정(Saerens/Elkan) | 460.2 → 434.2 (악화). **oracle**(2024 실제 평균을 그대로 사용)조차 328.6로 더 악화 | 문제가 단순 base-rate 이동이 아님 → concept drift로 판단 |
| 레짐 분리 가설 검증 | 2024년 내부만 시간분할(3~7월 학습→8~10월 검증)해 "판정방식 변화(ABS 등)" 가설 테스트 | 194.5 (cross-regime 460.2보다 더 나쁨) | 레짐 전환 가설 기각. 매년 특정 사건이 아니라 상시적인 forward-time 어려움 |
| 하이퍼파라미터 정규화 강화 | num_leaves/min_data_in_leaf/lambda_l2를 기존 대비 훨씬 강하게 재탐색 | 작은 표본(within-2024)에서는 최강 정규화가 이겼으나(194.5→233.8), 큰 표본(cross-regime)에서는 기존 튜닝값이 더 나음(460.2 vs 435.5) | 기존 하이퍼파라미터 튜닝은 (우려와 달리) 크게 잘못되지 않았음. 작은 표본 결과는 노이즈였음 |
| 트랙맨 팀 레벨 피처 | team_id↔trackman 팀코드 매핑(아래 참고)으로 팀 단위 구속/회전수/무브먼트/구종비율 프라이어 추가 | 460.2 → 429.9 (악화). 11개 신규 피처 전부 91개 중 하위 1/3 importance | 팀 단위(10개 그룹)는 정보량이 너무 낮아 노이즈로 작용. 개인 단위가 필요할 것으로 추정 |

### 트랙맨 연결: "0% 매칭"은 팀 레벨에서는 틀렸음 (단, 팀 레벨 자체는 유용하지 않았음)
기존 `USE_TRACKMAN=False` 판단 근거(`pitcher_id`↔`pitcher_trackman_id` 직접 조인 매칭률 0%)는
선수 개인 ID 조인만 시도한 결과. `pitcher_hand` 좌투수비율을 팀×시즌 단위로 매칭(`scipy.optimize.
linear_sum_assignment`)한 결과 `pitcher_team_id`(12~21)가 trackman `pitcher_team` 코드와 사실상
유일하게 대응됨을 확인(매칭 비용 대부분 <0.003, 실제 트랙맨 조인 시 매칭률 99.7%로 재확인):

12=DOO_BEA(두산), 13=LG_TWI(LG), 14=KIW_HER(키움), 15=LOT_GIA(롯데), 16=KIA_TIG(기아),
17=HAN_EAG(한화), 18=SAM_LIO(삼성), 19=NC_DIN(NC), 20=KT_WIZ(KT), 21=SK_WYV/SSG_LAN(SSG)

다만 위 표에서 보듯 팀 레벨 피처 자체는 forward 검증에서 손해였음. 한 팀·시즌 내 투수별
총 투구수를 정렬하면 상위 워크로드 투수들의 순위·손잡이가 train/trackman 간 거의 1:1로
정렬되는 것을 확인했음(예: 2019 두산 상위 5명 투구수 [2982,2646,2584,2350,2075] ↔
trackman [2887,2580,2414,2270,1973]) — **선수 개인 단위 매칭(팀×시즌 내 workload+손잡이
기반 Hungarian 매칭)은 아직 시도되지 않은, 가장 유망해 보이는 다음 후보**.

### 이후 추가 검증 (같은 날, 확장 윈도우 도입)

단일 분할(2019-2023→2024)로는 노이즈 여부 판단이 어려워, **expanding-window walk-forward
검증**(`code/walkforward_harness.py`)으로 전환: fold별로 2019→2020, 2019-2020→2021,
2019-2021→2022, 2019-2022→2023, 2019-2023→2024를 전부 예측해 5개 폴드 평균±표준편차로 판단.

**현재 기본 설정의 실제 변동성**: mean=706.0, **std=730.9** (표준편차가 평균만큼 큼).
연도별 score: 2020=0.0, 2021=1189.2, 2022=1880.8, 2023=0.0, 2024=460.2. 2020/2023이 0점인
이유는 그 시점까지의 누적 성공률과 실제 그 해 성공률의 격차가 유독 컸기 때문(-3.2%p, -3.95%p vs
다른 폴드 -1.4~-1.6%p) — 이미 확인한 "누적 대비 급락에 취약함" 패턴과 일치. **786점(실제 제출)은
이 분포의 정상 범위 안에 있음** — "운 나쁜 붕괴"가 아니라 이 파이프라인의 평범한 성능.

| 재검증 대상 | 방법 | 5-fold 평균 결과 | 판정 |
|---|---|---|---|
| 트랙맨 선수 개인 피처(위 표의 +21.1은 2024 단일폴드였음) | walk-forward 5폴드 재확인 | 706.0→701.4 (-4.6), 5폴드 중 2개만 개선, 1개는 -31.1 | **거짓양성으로 판명** — 2024에 국한된 우연이었음 |
| 개인맞대결/클러치 피처 ON | walk-forward 홀드아웃 빌드 시도 | **에러로 실행 자체가 안 됨** | `build_matchup_features`/`build_clutch_features`가 라벨 있는 데이터에서만 동작하고 test.csv용 정적 lookup 구현이 애초에 안 돼 있었음(코드 주석의 TODO) — 세션1 기각을 더 강하게 뒷받침 |
| **팀/팀맞대결/시즌추세 피처 OFF (3개 동시)** | walk-forward 5폴드, 2-fold 스크리닝 후 5-fold 확인 | **706.0 → 903.6 (+197.6, +28%)**, 5폴드 중 4개 개선(2020: 0→286.6) | **세션1 최대 발견 뒤집힘** — 세션1이 랜덤fold 기준 "채택"(+0.10%,+0.02%)했던 게 실제로는 forward 예측에 방해였음. **채택 권장(피처 끄는 쪽으로 변경)** |

### LGB+MLP 스태킹 재검증 (walk-forward, 개선된 피처 설정 기준)

팀/팀맞대결/시즌추세 피처 OFF 상태에서 `code/walkforward_stacking_check.py`로 LGB+MLP 2way
스태킹을 5-fold walk-forward 전체로 재검증. **트랙맨과 달리 이번엔 진짜 효과로 확인됨**:

| 단계 | 5-fold 평균 score |
|---|---|
| 원래 baseline(전체 피처, LGB 단독) | 706.0 |
| + 팀/팀맞대결/시즌추세 피처 제거(LGB 단독) | 903.6 (+197.6) |
| + MLP 스태킹 추가 | **918.1 (+14.5, 누적 +212.1 = 원래 대비 +30%)** |

폴드별: 2020 +39.1, 2021 -6.4, 2022 +14.1, 2023 +0.0(구조적 붕괴, 어떤 개선도 안 통함), 2024 +25.6
(5개 중 3개 개선). LGB-MLP OOF 상관계수도 폴드마다 0.848~0.876으로 세션1이 확인한 0.866과
비슷하게 나와 "다양성 있는 모델 스태킹" 논리 자체는 forward 기준으로도 유효함이 확인됨.

**다음 세션 실제 적용 목표**: `code/train_ensemble.py`의 `build_features` 기본값을
`use_team_feature=False, use_team_matchup_feature=False, use_season_trend_feature=False`로
변경(스태킹 구조는 유지) → 새 `submit.zip` 생성 전에 가능하면 LGB+RF, LGB+RF+MLP 3way 등
추가 조합도 같은 walk-forward로 검증 예정.

### RF 조합 재검증 (walk-forward, 개선된 피처 설정 기준)

`code/walkforward_stacking_check2.py`로 RF 단독/LGB+RF/LGB+RF+MLP 3way를 5-fold walk-forward로 확인:

| 조합 | 5-fold 평균 |
|---|---|
| LGB 단독 | 903.6 |
| RF 단독 | **993.3** (모든 조합 중 최고 평균) |
| LGB+RF | 966.9 |
| LGB+MLP | 918.1 |
| **LGB+RF+MLP 3way** | 934.0 |

RF 단독 평균이 제일 높지만, 폴드별로 보면 **fit 데이터가 적은 초기 폴드(2020 fit=23.7만행,
2021 fit=48.2만행)에서만 RF가 압도적**(배깅 기반이라 데이터 부족 시 부스팅보다 덜 과적합하는 것으로
추정)이고, 실제 제출 규모(147만행)에 가까운 데이터 많은 폴드(2022 fit=72.9만행, 2024 fit=122.2만행)
에서는 RF 우위가 사라지거나 역전됨. 이 폴드들에서 가장 일관되게 좋은 **LGB+RF+MLP 3way(2022 최고,
2024 두번째)를 실제 배포에 더 가까운 선택으로 채택**. LGB+RF 2way는 RF 단독보다도 못해 제외.

### 세션 2 종합 (원래 baseline 대비)

| 단계 | walk-forward 5-fold 평균 |
|---|---|
| 원래 baseline (전체 피처, LGB 단독) | 706.0 |
| + 팀/팀맞대결/시즌추세 피처 제거 | 903.6 |
| + LGB+MLP 스태킹 | 918.1 |
| **+ RF 추가(LGB+RF+MLP 3way)** | **934.0 (원래 대비 +228.0, +32%)** |

**다음 세션 실행 항목**: `code/train_ensemble.py`에 위 설정(피처 3개 OFF + LGB+RF+MLP 3way 스태킹)을
실제 반영하고 새 `submit.zip`을 생성해 제출.

### XGB/CAT 조합 재검증 (walk-forward) — session1 판단이 그대로 확인됨

`code/walkforward_stacking_check3.py`로 XGB/CAT 단독 및 LGB와의 조합을 5-fold walk-forward로 확인:

| 조합 | 5-fold 평균 | LGB와의 OOF 상관계수 |
|---|---|---|
| LGB 단독 | 903.6 | - |
| XGB 단독 | 903.6 | 0.967~0.974 |
| CAT 단독 | 881.3 | 0.939~0.955 |
| LGB+XGB | 905.4 (+1.8) | |
| LGB+CAT | 908.3 (+4.7) | |
| LGB+XGB+CAT 3way | 914.2 (+10.6) | |
| **LGB+RF+MLP 3way (최종 채택)** | **934.0 (+30.4)** | RF 0.933~0.963 / MLP 0.848~0.876 |

LGB-XGB/CAT 상관계수(0.94~0.97)가 LGB-RF(0.93~0.96)나 LGB-MLP(0.85~0.88)보다 훨씬 높아 스태킹
이득이 작음 — session1이 랜덤fold 기준으로 내렸던 "같은 부스팅 계열이라 상관관계 높아 손해"
판단이 walk-forward 기준으로도 그대로 확인됨. XGB/CAT을 LGB+RF+MLP에 추가로 얹는 5-way는
이미 낮은 개별 기여도(914.2 << 934.0)로 미루어 볼 때 도움이 안 될 게 명확해 테스트 생략.

**최종 결론: 검증된 조합 중 LGB+RF+MLP 3way(934.0)가 최선. 여기서 앙상블 조합 탐색 종료.**

### 아직 재검증되지 않은 것
- LGB vs LR는 이미 확인(LGB가 붕괴율도 낮음: 5.78x vs 12.72x, 단일폴드 기준).
- Recency-weighted 학습(최근 시즌에 sample weight): 단일폴드(2019-2023→2024) 기준 감쇠
  강할수록 악화(436.6→350.9)로 기각. walk-forward 전체로는 재확인 안 함.
- 딥러닝 확장 방향(domain generalization 기법 IRM/Group DRO, 선수이력 시퀀스모델, 그래프 임베딩
  매치업, FT-Transformer/TabNet, 보정인식 손실함수)은 논의만 하고 구현/검증은 안 함 — 세션 대화 기록 참고.
