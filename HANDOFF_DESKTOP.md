# 데스크탑 이어서 진행하기 (2026-08-20 세션 인계 문서)

이 노트북에서 RF(RandomForest) 학습이 메모리 부족으로 계속 멈춰서, 더 사양 좋은
데스크탑에서 이어서 진행하기 위한 인계 문서. 아래 순서대로 진행하면 됨.

## 1. 지금까지 확정된 것 (재검증 불필요, 그대로 신뢰)

- **버그가 아니라 하드웨어 한계였음**: 이 노트북 총 메모리 15.6GB, 작업 중 여유
  2~4GB. `python train_ensemble.py` 실행 시 LGB(5분 내외/fold)·MLP는 정상 완료했지만
  RandomForest(`train_rf`, n_estimators=400/max_depth=14, n_jobs=-1) 단계에서
  CPU 사용률이 거의 0으로 떨어지고(스와핑 추정) 1시간+ 진행이 안 됨. 두 번(2024년
  1개 폴드 subset 테스트, 전체 147만행 production 학습) 모두 같은 증상으로 강제
  종료함. `LOKY_MAX_CPU_COUNT` 환경변수로 코어 제한을 시도했지만 효과 없었음 —
  sklearn RandomForest는 기본적으로 threading 백엔드를 쓰기 때문에 loky(프로세스)용
  환경변수가 적용 안 되는 것으로 추정.
- **피처/모델 설정 자체는 walk-forward로 이미 검증 완료**(랜덤 K-fold 아님, 시간순
  홀드아웃 기준 — 이 대회는 test.csv가 train.csv에 없는 미래 시즌(2025)이라 랜덤
  fold 검증은 신뢰할 수 없음이 이미 확인됨, 상세: `README.md` "세션 2"):
  - `use_team_feature`/`use_team_matchup_feature`/`use_season_trend_feature` 3개
    OFF: walk-forward 5-fold 평균 706.0 → **903.6 (+197.6, +28%)**
  - LGB+MLP 2-way 스태킹 추가: 903.6 → **918.1 (+14.5)**
  - LGB+RF+MLP 3-way 스태킹(RF는 subset 규모 데이터로는 학습 성공): 918.1 →
    **934.0 (+15.9, 원래 대비 총 +32%)**
  - 트랙맨 "일관성"(표준편차) 피처: LGB 단독 기준 약한 순긍정(+9.6, 5폴드 중 3개
    개선)이지만 3-way 스택에서는 RF 문제로 검증 못 함 → **이번엔 미채택**
    (`USE_TRACKMAN=False` 유지). 상세: 아래 4번 항목.
- **코드는 이미 반영 완료**: [code/train_ensemble.py](code/train_ensemble.py)의
  `build_features` 기본값 3개 False로 변경, `main()`을 LGB+RF+MLP 3-way 스태킹으로
  변경(RF 추가, bundle에 `rf_models` 포함). [open/baseline_submit/script.py](open/baseline_submit/script.py)도
  RF 추론 + 3-way 스태킹으로 동기화 완료. **이 두 파일은 이미 최종 상태 — 추가 수정 불필요.**
- **안전한 백업 확보됨**:
  - `open/1차_submit.zip` — 기존 LGB+MLP 2-way로 실제 리더보드 **786점** 받은 검증된
    제출 파일 (예전 `open/submit.zip`을 리네임한 것).
  - `code/model/ensemble_bundle_2way_backup.pkl` — 위 786점 제출에 쓰인 2-way 모델
    번들 백업. 새 3-way 학습이 또 실패해도 이걸로 되돌릴 수 있음.

## 2. 데스크탑에서 옮겨야 할 것

git에는 `open/data/`(대용량 CSV)와 `*.pkl` 모델 파일이 `.gitignore`로 빠져있어서
git만으로는 안 옮겨짐. 아래를 수동으로 복사(USB/공유폴더/클라우드 등):

- `open/data/train.csv` (352MB), `trackman_history.csv`, `test.csv`, `sample_submission.csv`
- (선택) `code/model/ensemble_bundle_2way_backup.pkl` — 데스크탑에서도 만약을 대비한 백업 필요하면

git으로 옮길 수 있는 것(단, 현재 로컬에 **커밋 안 된 상태** — 필요하면 커밋 후 push/pull,
또는 폴더째 복사):
- `code/train_ensemble.py`, `open/baseline_submit/script.py` (수정됨, `git diff`로 확인 가능)
- `code/trackman_pitcher_lib.py`(신규 함수 `build_trackman_consistency_prior`/
  `attach_trackman_consistency_features` 추가), `code/walkforward_*.py`(신규 검증
  스크립트 다수), `code/build_submission_zip.py`(신규, 아래 4번에서 사용)

## 3. 데스크탑에서 할 일 (순서대로)

```bash
# 0) 환경: requirements.txt와 동일한 버전 설치 확인
#    lightgbm==4.3.0, torch==2.7.1, scikit-learn==1.8.0, pandas==2.1.4,
#    numpy==1.26.4, joblib==1.5.3 (open/baseline_submit/requirements.txt 참고)

# 1) 전체 데이터로 3-way 앙상블 프로덕션 학습 (여기서 막혔던 부분 - 메모리 넉넉하면
#    별도 옵션 없이 정상 완료될 것으로 예상. 그래도 느리면 아래 "막히면" 참고)
cd code
python train_ensemble.py
#   -> code/model/ensemble_bundle.pkl 생성 확인 (기존 파일을 덮어씀 -
#      혹시 모르니 실행 전 ensemble_bundle.pkl을 한번 더 백업해두는 것도 안전)

# 2) 제출 zip 생성 (script.py + requirements.txt + 방금 만든 model bundle을 묶음)
python build_submission_zip.py
#   -> open/submit.zip 생성. 기존 open/submit.zip이 있으면 자동으로
#      open/submit_prev.zip으로 백업 후 새로 만듦.

# 3) 제출 전 반드시 추론 스모크 테스트 (RF를 추론 경로에 추가한 뒤 한 번도
#    실제로 끝까지 돌려본 적이 없음 - 여기서 에러 나면 제출 전에 잡아야 함)
cp model/ensemble_bundle.pkl ../open/baseline_submit/model/ensemble_bundle.pkl
cd ../open/baseline_submit
python script.py
#   -> output/submission.csv 생성 확인. row 수가 data/test.csv와 같은지,
#      확률값이 [0,1] 범위인지, NaN 없는지 확인.
```

**막히면(그래도 RF가 느리거나 메모리 이슈면)**: `code/train_ensemble.py`의
`train_rf` 함수(파일 내 검색) 안 `RandomForestClassifier(...)`의 `n_jobs=-1`을
`n_jobs=4`나 `n_jobs=2`로 바꿔서 재시도. 하이퍼파라미터가 아니라 병렬도 설정이라
모델 결과에는 영향 없음(속도만 달라짐).

## 4. 시간 여유 있으면: 트랙맨 "일관성" 피처 3-way 재검증

이 노트북에서 RF 문제로 못 끝낸 부분. 데스크탑이 여유 있으면 아래로 결론을 낼 수 있음:

```bash
cd code
python walkforward_trackman_consistency_stack_check.py
# 5개 시즌(2020~2024) x 2가지 설정(트랙맨 유/무) x LGB+RF+MLP 3-way를 전부 학습
# → 마지막 SUMMARY의 "mean delta"가 뚜렷한 양수(+)면 채택 검토, 애매하거나 음수면
#   기존 결론(미채택) 유지. 자세한 배경은 코드 상단 주석과
#   code/trackman_pitcher_lib.py의 build_trackman_consistency_prior 주석 참고.
```

이게 오래 걸리면 `code/walkforward_trackman_consistency_stack_check_light.py`
(target_season=2024 한 폴드만) 먼저 돌려서 방향성만 빠르게 확인해도 됨.

## 5. 그래도 안 되면 (최후 수단)

RF가 데스크탑에서도 안 되면 3-way를 포기하고 2-way(LGB+MLP)로 제출:
- `code/train_ensemble.py`의 `main()`에서 RF 관련 3줄(`train_rf` 호출,
  calibrators/calibrated 루프의 `"rf"` 항목, `meta_X`의 `calibrated["rf"]`)과
  `bundle`의 `"rf_models"` 항목을 빼면 2-way로 되돌아감(git diff로 원래 코드 확인 가능).
- 또는 그냥 `code/model/ensemble_bundle_2way_backup.pkl`을
  `code/model/ensemble_bundle.pkl`로 복사해 쓰고 `build_submission_zip.py` 실행
  (단, 이건 피처 3개 제거 개선 전 버전이므로 706.0 기준 — 권장 안 함. 코드 되돌리고
  재학습하는 게 나음).
</content>
