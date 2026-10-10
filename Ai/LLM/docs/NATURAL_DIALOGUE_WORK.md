# 자연스러운 대화 개선 작업 결과 · 2026-10-10

## 저장소와 작업 위치

- 계정: `KORemc2` (GitHub 앱 및 Windows Credential Manager 확인).
- 원본: https://github.com/HealthyMon/healthmon-capstone
- Fork: https://github.com/KORemc2/healthmon-capstone
- GitHub REST의 `fork=true`, `parent`와 `source`로 Fork 관계를 확인했다.
- 원본/Fork `main`: `47b0c308629ce4bc1b9f78430d13007f052ce31b`.
- Clone: `E:\qwen-llm\HEALTHMON`; LLM: `Ai/LLM`.
- 기존 `E:\qwen-llm`은 Git 저장소가 아니었다. Fork의 `Ai/LLM`은 `.gitkeep`만 있었으므로 기존 구현을 별도 경로로 가져왔다. 기존 파일을 덮어쓰지 않았다.
- 현재 원격은 사용자 Fork인 `origin`만 있다. 사용자 최신 지시에 따라 `upstream` 연결은 제거했고, `remote.pushDefault=origin`으로 설정했다. 원본 전송은 나중에 사용자가 명시적으로 요청할 때만 진행한다.
- 초기 Push는 공개 파일 검증 문제로 자동 승인 검토에서 차단되었다. 실제 로컬 비밀값과 key 패턴 검사, 합성 데이터 출처 검증 후 재시도 승인을 받았다. Windows의 여러 자격증명 중 `credential.https://github.com.username=KORemc2`를 지정해 실제 Push를 완료했다. 이 선택은 해당 Clone의 Git 설정에 저장했다.

## 복구 기준선

`41e21eb70687f15d9d5a6b925b972b45355a4d20`에 기존 코드·설정·합성 데이터 204파일을 보존했다. `baseline/local-20261010` 브랜치와 `baseline/llm-local-20261010` 태그가 Fork에 존재한다. 원본 초기 커밋은 `baseline/github-20261010` 브랜치와 태그로 보존했다.

모델과 운영 데이터는 원격 코드 백업과 별도로, `E:\qwen-llm\backups\natural-baseline-20261010`에 복사했다. 2,992파일 / 24,303,647,199바이트이며 파일마다 원본과 복사본 SHA256을 대조했다. `manifest.json`이 복구 검증 기준이다. 이 백업에는 비밀값이 있어 공개 저장소로 전송하지 않았다.

- 기반 모델: `E:\qwen-llm\models\qwen3.5-9b`.
- 현재 보존 Baseline: `E:\qwen-llm\runs\expanded-20260929\checkpoint-epoch-1`.
- 이전 파일럿: `E:\qwen-llm\runs\pilot-20260929\checkpoint-epoch-3`.
- 기존 LoRA 9개와 optimizer 체크포인트, runs의 학습 설정·로그·실제 평가 자료를 모두 복사했다.
- 데이터 `pilot_v1/v2`, `improvement_v1/v2/v3`를 원본 그대로 보존했다.
- PostgreSQL은 백업 전 `postmaster.pid`가 없었고, 중지된 `runtime/pgdata`를 복사했다. Clone에는 별도 새 pgdata와 새 비밀값으로 테스트 DB를 생성했다.
- Docker 데몬은 연결되지 않았다. 컨테이너 내부 모델이나 볼륨은 조사하지 못했으며, Docker 백업이 완료됐다고 보지 않는다. 현재 확인한 모델은 Windows 로컬 파일이다.

보존된 모델을 안정 운영 승인 모델이라고 주장하지 않는다. 기존 serving 설정도 memory/quest 회귀와 사람 검수 미완료를 명시했다.

## 코드 구조 확인과 개선

Qwen3.5-9B BF16 및 PEFT LoRA는 `src/model.py`, 기존 학습은 `scripts/train_lora.py`다. 7축 legacy 성격·감쇠·근거 소비·Memory 검색/정정/삭제·Agent·FastAPI 채팅과 웹 연동이 이미 있었다. Persona Lab은 별도 4축 시스템이고, Character Lab은 Activity / Affinity(`affection`) / Initiative / Mood와 보존 legacy 축, 감정·의도 Analyzer, 구조화 응답 검수·Export를 제공한다. 이들을 하나의 동일한 성격 시스템으로 오인하지 않았다.

기존 원시 평가에는 반복 응답, 상투적인 공감, 말투 혼합, 최신 기억 누락, 퀘스트 수치 오류가 남아 있었다. 현재 수집한 출력에서도 사용자 발화 재진술, 관련 없는 휴식 권유, 구체성을 잃는 반응과 근거 없는 감정 추정이 확인됐다.

이번 변경은 기존 프롬프트를 `baseline`으로 유지하고 `natural-v1`을 선택할 때만 일상 반응·질문 필요성·응원 빈도·주제 연결 지침을 추가한다. 최근 대화와 Memory 내용·ID는 양쪽에 동일하게 전달한다. 검증·성격 계산·퀘스트 실행 권한은 기존 함수를 재사용한다. `src/rules.py`, `persona_engine.py`, `character_lab_engine.py`와 성장 규칙 설정은 수정하지 않았다.

`model_registry`는 모델 상태·선택·복구와 삭제 후보 목록을 관리한다. Baseline/Selected/공유 adapter 경로는 보호하며 실제 파일을 삭제하지 않는다. 모델별 비교는 하나의 GPU batch lock 안에서 같은 문맥·seed·생성 설정을 사용하고, 오류가 나도 원래 어댑터와 프롬프트를 복원한다. JSON 계약과 fallback은 기존 `respond()`를 사용한다.

기존 Lab에 **05 모델 비교 · 평가**를 추가했다. 원시 응답·수정 시도·fallback과 모델별 답변을 나란히 보여 준다. 자연스러움·성격·멀티턴·반복·Memory·정체성·구조화 분석·호환성의 8항목 점수 및 선호 모델을 저장한다. 미평가를 허용하고 자동 채택하지 않는다. Rejected 모델도 개발 비교는 가능하지만 기본 serving으로 선택할 수 없다.

## 데이터와 실험

기존 최신 데이터는 학습216/검증43/기존 시험58건이며 외부 데이터 0건, 사람 검수 0건이었다. 현재 원본 JSONL/manifest와 이전 감사 자료를 확인했다. 기존 시험은 이미 공개된 개발 회귀셋으로 취급한다.

새 `natural_sft_v1`에는 개별 작성한 합성 예제16건을 추가했다. 12건은 학습, 4건은 검증에 배치하고 기존 성격 학습 예제4건을 학습에 보존해 총16/4건이다. 원본 데이터는 덮어쓰지 않았다. 실제 tokenizer 검사에서 학습 최대1,301토큰, 검증 최대1,276토큰이며 설정1,536 안에서 잘리지 않는다. 별도 `natural_eval_v1`은 16개 고정 사례다. 동일 제작자의 소규모 합성 회귀셋이며 독립 성능 벤치마크가 아니다. 외부 데이터셋·유료 외부 LLM API·대량 자동 생성은 사용하지 않았다. 외부 데이터의 라이선스나 다운로드 가능성을 검증했다고 주장하지 않는다.

| 실험 | 실제 수행 | 판정 |
|---|---|---|
| Baseline | 기존 expanded epoch1 보존, 16사례 실제 재평가 | 현재 선택 유지. 운영 승인 없음 |
| V1 | 동일 어댑터에서 프롬프트만 변경, 16사례 실제 비교 | Rejected. 전반적 개선 불충분, 어색한 표현·근거 없는 잘못 추정 |
| V2 smoke | 기존 어댑터에서 BF16 LoRA 추가 SFT 1epoch, 16train/4val, optimizer 2회와 16사례 실제 비교 | Rejected. 상투적 반응·근거 없는 추정·설명 부족 지속 |
| V3 | 독립 브랜치와 멀티턴·성격 혼합 실험 전제조건 작성 | 학습·평가 미실행, 체크포인트 없음 |

V1은 Baseline/V1 모두 fallback0/16, 질문 포함2/16→1/16, 퀘스트 행동0/16이었다. 이를 자연스러움 우승 점수로 해석하지 않는다. 예를 들어 속상함에 대한 V1의 `네가 잘못한 건 아니겠지`는 근거 없는 잘못 추정이며, 상세 설명 요청에서도 충분한 설명을 하지 못했다. 수영·오후 선호 근거를 전달한 시험 결과만으로 일반 Memory 정확성을 보장하지 않는다. Analyzer 호환성 예제2개는 별도 분류 결과로 기록했다.

V2의 실제 학습 커밋은 `6c6484c6368c76b066715ca44d2335f1b8aeb080`이다. 기반 모델 revision은 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, r8/alpha32/lr2e-5/batch1/accum8/seed42/max1536이다. 검증 loss는 `0.38715720549225807`이고 저장 후 다시 로드한 loss도 같았다. GPU 최대 할당22.362GiB/예약22.887GiB였다. 서로 다른 검증 데이터의 loss를 기존 모델 우승 근거로 비교하지 않는다. 새 가중치 경로는 `runs/natural-v2-smoke-20261010/checkpoint-epoch-1`이며 기존 가중치를 덮어쓰지 않았다. 세부 설정·실제 이벤트는 V2 브랜치의 `experiments/results/natural-v2-training.json`에 있다.

V2도 16사례에서 fallback0, 질문 포함1, 퀘스트 행동0이었다. 메일을 오래 쓴 이유를 `잠든 건가 봐`라고 추정하고, 말다툼에 `네 잘못이 아니야`라고 단정했다. 따라서 현재 기본 모델을 교체할 근거가 부족하다. V2 대화 결과는 같은 문맥 해시·seed·설정을 검증한 V1의 동결 Baseline 출력과 비교했다. 두 평가의 Analyzer 예제는 Baseline 어댑터만 검사했으며 V2 Analyzer 전이 능력은 검증하지 않았다. Rejected V2 가중치는 미사용 삭제 후보로만 분류했고 실제로 삭제하지 않았다. V1은 Baseline과 동일한 어댑터 경로이므로 삭제 후보가 아니다.

## 검증·정리·제한

격리된 실제 PostgreSQL17.11/pgvector0.8.2/BGE-M3 CPU를 포함한 Python 테스트456개를 통과했고, GPU 잠금 테스트1개도 GPU 작업과 분리해 통과했다. 합계457개이며 Node 웹 API 테스트5개도 통과했다. DB 경계 테스트의 생성기는 controlled backend이므로 실제 Qwen 품질 평가와 구분한다. 실제 Qwen 응답은 V1의 32응답과 별도 Analyzer 시험에 기록했다. 처음 DB 미시작 또는 GPU 평가와 동시에 실행한 검사에서는 환경 충돌로 실패했고, 환경을 준비하고 분리한 위 결과로 재검증했다.

모델·데이터·DB·학습 산출물은 삭제하지 않았다. `.gitignore`에 추가한 가중치·optimizer·SQLite·백업·로그·환경파일 규칙으로 공개 업로드를 방지했다. 삭제 후보는 참조를 확인해 목록으로만 제시한다. 역사 문서·출처·라이선스를 삭제하지 않았다. 캐시 정리는 Clone 내부의 검증된 캐시 경로에 한정한다.

## 복구와 원격 변경

현재 선택은 `baseline-expanded`다. 다른 후보를 시험한 뒤 복구하려면 아래 명령을 실행하고 서버를 재시작한다.

```powershell
Set-Location E:\qwen-llm\HEALTHMON\Ai\LLM
E:\qwen-llm\.venv\Scripts\python.exe scripts\model_registry.py --select baseline-expanded
E:\qwen-llm\.venv\Scripts\python.exe scripts\serve.py --adapter runs/expanded-20260929/checkpoint-epoch-1
```

기존 환경 자체는 `E:\qwen-llm`에 그대로 있다. 코드 전체를 기준선으로 확인하려면 기존 브랜치를 강제 변경하지 말고 Clone 루트에서 별도 worktree를 만든다.

```powershell
git worktree add E:\qwen-llm\HEALTHMON-baseline baseline/llm-local-20261010
```

해당 worktree의 `Ai/LLM`에 모델·어댑터·환경 경로를 연결해 실행한다. 모델 복구는 백업의 같은 상대 경로를 새 위치에 복사하고 `manifest.json`의 SHA256과 대조한다. DB는 백업 복사본을 새 pgdata 경로에서 시작하고 복구한 로컬 자격증명을 사용한다. 실행 중인 기존 pgdata를 덮어쓰지 않는다.

Fork의 baseline 브랜치·태그 및 `feat/llm-natural-dialogue`, `exp/llm/natural-v1/v2/v3`에 실제 Push를 수행했다. 실험 가중치는 로컬에만 보존한다. 원본에 직접 Push하거나 자동 Merge하지 않았다. 앞서 생성한 원본 Draft PR #2는 사용자 지시에 따라 2026-10-10 18:10 KST에 닫았다. REST 확인 결과 `state=closed`, `merged=false`다. GitHub에는 닫힌 PR 이력이 남는다. 채팅의 PR 첨부와 임시 생성 스크립트는 제거했다. 이후 작업·커밋·Push는 사용자 Fork에만 한정하며, 원본 PR 생성은 현재 수행 범위에서 제외한다.

브라우저에서 `http://127.0.0.1:8002/`를 실제 실행하고 Baseline/V2의 동일 Memory 비교도 확인했다. 양쪽 모두 수영 선호를 답하고 fallback 없이 각각1회 생성했으며, 별도 로컬 평가 파일이 저장됐다. 사람 평가 점수는 제출하지 않았다. 서버는 개발 비교를 위해 실행 상태로 두었다. 기존 베타 DB와 분리된 테스트 DB도 실행 중이다.

데이터와 원시 평가 JSON은 Git의 줄바꿈 변환을 막아 저장 시 SHA256과 Clone 후 바이트가 일치하게 했다. `scripts/check_experiment_integrity.py`로 기존 최신 데이터3파일·새 학습/검증2파일·고정 사례1파일을 실제 확인했다. 저장소 추적파일의 실제 로컬 비밀값·키 패턴 검사도 0건이었다.
