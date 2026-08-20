# build_submission_zip.py
# ------------------------------------------------------------
# code/train_ensemble.py 학습이 끝나 code/model/ensemble_bundle.pkl이 생성된 뒤,
# 대회 제출용 zip(script.py + requirements.txt + model/ensemble_bundle.pkl)을
# 만든다. open/baseline_submit/의 script.py/requirements.txt를 그대로 쓰고,
# model만 방금 학습한 번들로 교체해서 담는다.
#
# 사용법: code/ 디렉토리에서 `python build_submission_zip.py`
#   (먼저 python train_ensemble.py로 code/model/ensemble_bundle.pkl을 만들어둘 것)
# ------------------------------------------------------------
import os
import shutil
import zipfile

BUNDLE_PATH = "./model/ensemble_bundle.pkl"
SCRIPT_PATH = "../open/baseline_submit/script.py"
REQUIREMENTS_PATH = "../open/baseline_submit/requirements.txt"
OUT_ZIP = "../open/submit.zip"


def main():
    for p in [BUNDLE_PATH, SCRIPT_PATH, REQUIREMENTS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"필요한 파일이 없음: {p}")

    bundle_size_mb = os.path.getsize(BUNDLE_PATH) / 1e6
    print(f"model bundle: {BUNDLE_PATH} ({bundle_size_mb:.1f} MB)")

    if os.path.exists(OUT_ZIP):
        backup = OUT_ZIP.replace(".zip", "_prev.zip")
        print(f"기존 {OUT_ZIP} -> {backup}로 백업")
        shutil.move(OUT_ZIP, backup)

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(SCRIPT_PATH, arcname="script.py")
        z.write(REQUIREMENTS_PATH, arcname="requirements.txt")
        z.write(BUNDLE_PATH, arcname="model/ensemble_bundle.pkl")

    out_size_mb = os.path.getsize(OUT_ZIP) / 1e6
    print(f"[OK] {OUT_ZIP} 생성 완료 ({out_size_mb:.1f} MB)")
    with zipfile.ZipFile(OUT_ZIP) as z:
        print("zip 내용:", z.namelist())


if __name__ == "__main__":
    main()
