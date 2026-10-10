# HEALTHMON LLM 실험 환경

기존 로컬 LLM을 `Ai/LLM`에 가져온 개발 환경입니다. 원래 실행 폴더 `E:\qwen-llm`은 보존했습니다. 과거 실행 기록은 `START_HERE.md`와 기존 감사 문서에 남아 있으며 현재 서버 상태를 뜻하지 않습니다.

현재 작업은 **KORemc2의 Fork 클론**에서만 진행합니다. `origin`은 사용자 Fork이며 원본 원격 연결은 제거했습니다. 원본 전송은 사용자가 나중에 명시적으로 요청할 때 진행합니다. 다른 PC에서는 실제 LLM 코드가 있는 브랜치로 클론하세요.

```powershell
git clone --branch feat/llm-natural-dialogue https://github.com/KORemc2/healthmon-capstone.git HEALTHMON
Set-Location HEALTHMON
git config core.hooksPath .githooks
git config remote.pushDefault origin
Set-Location Ai\LLM
```

V1–V4 독립 실험의 자료 출처·실행 조건은 [한국어 대화 실험 기록](docs/EXTERNAL_DIALOGUE_EXPERIMENTS.md)에 있습니다. 외부 원문·변환 데이터·가중치는 로컬 연구용으로 보관합니다.

## 실행

현재 PC에서는 이 디렉터리에서 아래 명령을 사용합니다.

```powershell
E:\qwen-llm\.venv\Scripts\python.exe scripts\serve_persona_lab.py --port 8002
```

`http://127.0.0.1:8002/`의 **05 모델 비교 · 평가**에서 같은 입력·성격·Memory·최근 대화·seed로 비교합니다. 원시 출력, 재시도, fallback을 표시합니다. 8개 항목은 직접 평가하며 미확인 항목은 미평가로 둡니다. 평가 저장만으로 모델을 채택하지 않습니다. 입력과 평가 결과는 Git에서 제외된 로컬 `runs/model-comparisons`에 저장되므로 합성 사례만 사용합니다.

다른 PC에서는 Python 환경을 설치하고 Qwen/BGE와 필요한 LoRA를 별도로 복원해야 합니다. 모델·DB·비밀값은 GitHub에 포함되어 있지 않습니다. `config/qwen3.5-9b.json`의 로컬 경로를 수정하고 `.env.example`을 참고해 새 `.env`를 만드세요. 현재 PC의 `models`, `.venv`, `runtime/pgsql` 연결은 이전 환경을 재사용하는 로컬 설정입니다.

## 실험과 선택

`config/model_registry.json`에서 모델 ID, 상태, 어댑터 경로와 프롬프트를 관리합니다. 기존 Baseline은 보존됩니다. 학습 가중치 변경 없이 사용하는 `natural-v1` 프롬프트도 독립 실험입니다. 기본 모델은 Baseline이며 자동 승격하지 않습니다.

```powershell
E:\qwen-llm\.venv\Scripts\python.exe scripts\model_registry.py
E:\qwen-llm\.venv\Scripts\python.exe scripts\model_registry.py --cleanup-inventory
# 복구: 기존 Baseline으로 선택을 되돌림
E:\qwen-llm\.venv\Scripts\python.exe scripts\model_registry.py --select baseline-expanded
```

다른 모델 선택은 실제 호환성 검사와 사람 검수 기록이 필요합니다. `Candidate`, `Rejected`, `Archived` 기록과 파일 삭제는 별개이며 자동 삭제 명령은 제공하지 않습니다. 전환 후 실행 중인 서버는 재시작하세요. 기존 채팅 서버도 명시적 `--adapter`, 환경변수, registry 순으로 어댑터를 결정하며 `--base`를 지원합니다.

## 재현

```powershell
E:\qwen-llm\.venv\Scripts\python.exe scripts\evaluate_natural.py --output runs/my-new-evaluation.json
E:\qwen-llm\.venv\Scripts\python.exe scripts\train_lora.py --run my-new-training --epochs 1 --max-length 1536 --data data/natural_sft_v1 --profile natural-v1 --init-adapter runs/expanded-20260929/checkpoint-epoch-1 --lr 0.00002
E:\qwen-llm\.venv\Scripts\python.exe -m pytest tests -q --tb=short
node --test tests/test_web_api.mjs
```

학습과 추론은 GPU 잠금으로 동시에 실행하지 않습니다. 재실행 시 새 출력 이름을 사용하세요. 실험별 Git 커밋·데이터 해시·하이퍼파라미터·경로는 `experiments`와 `runs/<실험>/config.json`에 기록합니다. 실제 결과와 복구 절차는 `docs/NATURAL_DIALOGUE_WORK.md`를 확인하세요.
