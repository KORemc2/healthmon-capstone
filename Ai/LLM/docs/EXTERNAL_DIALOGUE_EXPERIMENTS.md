# 한국어 대화 독립 실험

작업 위치는 사용자 Fork를 클론한 `E:\qwen-llm\HEALTHMON\Ai\LLM`이다. 현재 Git 원격은 KORemc2 Fork의 `origin`만 있으며 원본 PR은 만들지 않는다. `.githooks/pre-push`는 이 Fork 이외 목적지의 Git Push를 차단한다.

## 자료와 사용 범위

- [Songys Chatbot_data](https://github.com/songys/Chatbot_data): 작성자가 만든 한국어 문답, MIT. 일상 label=0에서 말투 자료를 추출한다. 원본 LICENSE와 README를 다운로드 경로에 보존한다.
- [KorEmpatheticDialogues](https://huggingface.co/datasets/passing2961/KorEmpatheticDialogues): EmpatheticDialogues의 DeepL 번역, CC BY-NC 4.0 및 연구 목적 명시. 로컬 비상업 연구만 수행한다. 원본·변환 데이터·학습 가중치는 공개하지 않는다. 상업 운영 허용 자료로 취급하지 않는다.
- [KREW Korean role-playing](https://huggingface.co/datasets/huggingface-KREW/korean-role-playing): Apache 2.0. [Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 기반 `general-roleplay-data`만 다운로드했다. 검사한 자료의 고정된 다른 캐릭터 정체성과 역할극 지시가 HEALTHMON과 충돌하므로 이번 학습에서는 제외한다. 커플·YouTube·EXA subset은 다운로드하지 않았다.
- AI-Hub [공감형 대화](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&currMenu=115&dataSetSn=71305), [페르소나 대화](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71302), [한국어 멀티세션](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71630)는 조사했지만 신청·인증을 진행하지 않았다. 사용 승인이 확인되지 않아 다운로드·학습에 사용하지 않았다.
- [NLPBada의 AI-Hub 변환본](https://huggingface.co/datasets/NLPBada/korean-persona-chat-dataset)은 업로더의 MIT 표시만으로 원출처 이용·재배포 권한까지 확인할 수 없어 제외했다. [Ammad1Ali 자료](https://huggingface.co/datasets/Ammad1Ali/Korean-conversational-dataset)도 로봇 같은 AI 자기소개와 불명확한 출처로 제외했다.

정확한 revision·파일 URL·SHA256은 `experiments/external-dialogue-sources.json`에 있다. 원문은 Git에서 무시하는 `data/external_raw/`, 학습용 변환본은 `data/external_sft_*/`에 있다. 외부 유료 API는 호출하지 않았다. Parquet 검사 도구 pyarrow 19.0.1은 Clone의 `runtime/dataset-tools`에만 설치했다.

## 비교 조건

네 실험 모두 `runs/expanded-20260929/checkpoint-epoch-1`에서 독립적으로 시작한다. 앞 실험의 가중치를 다음 실험에 이어 학습하지 않는다. BF16 PEFT LoRA r8/alpha32, lr2e-5, seed42, batch1/누적8, 1epoch, 최대3072토큰을 사용한다. 기존 Windows 학습기를 재사용한다.

각 학습셋은 새 자료32건과 기존 성격·정체성·Memory 예제8건, 검증8건, 시험8건이다. 이는 실행 가능성을 확인하는 작은 실험이며 완전 학습·성능 입증을 의미하지 않는다. Character 응답에는 실제 serving의 `persona_messages`, 분류에는 별도 Analyzer 프롬프트를 사용한다. 기존 학습기의 기본 legacy 동작은 유지한다.

처음 전체 vocabulary logit을 만든 점검은 GPU 24,245/24,576MiB 상태에서 첫 backward가 끝나지 않아 중단했다. 실패 기록을 보존했다. 새 `--assistant-logits-only` 옵션은 정답 토큰 바로 앞의 logit만 계산한다. 기존 masked causal loss와 손실값·기울기가 같은지 테스트했다. 이후 실제 BF16 8건/검증2건 점검은 optimizer1회, 저장·재로드 검증 손실 `0.8835637271404266`, 최대 할당20.477GiB로 성공했다. 이전 어댑터와 학습 기본값은 변경하지 않았다.

| ID / 브랜치 | 새 자료32건 |
|---|---|
| tone-v1 / exp/llm/tone-v1 | 일상 한국어 문답 |
| emotion-v2 / exp/llm/emotion-v2 | 감정 대화24 + 기존 schema에 맞춘 독립 분류 예제8 |
| mixed-v3 / exp/llm/mixed-v3 | 일상12 + 감정12 + 분류4 + 성격 수치 조건4 |
| multiturn-v4 / exp/llm/multiturn-v4 | 실제 다중 턴24 + 별도 Memory 근거8 |

공개 원본의 감정 label은 사용자 감정의 출처 메타데이터다. 이를 캐릭터 Mood 값으로 복사하지 않는다. 성장 계산과 스탯 변경 권한은 서버 규칙에 남긴다.

원본의 대화 단위와 공식 감정 데이터 split을 유지하고, 일상 문답은 질문 hash로 분리한다. 동일 pair와 충분히 긴 동일 발화가 split을 넘으면 제외한다. 평가 사례와 같은 사용자 문장도 제외한다. 전화·주민번호·이메일·URL 패턴, 부적합 표현, 가짜 자기 경험, 상투적 응원 및 기존 출력 계약 위반을 필터링한다. 길이는 실제 tokenizer로 확인하고 자르지 않는다. 자동 필터가 모든 개인정보·의미 중복·품질 문제를 해결했다고 주장하지 않는다. 사람 검수는 아직 없다.

재현 명령:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_dialogue_sources.py
.\.venv\Scripts\python.exe scripts\prepare_dialogue_experiments.py
.\.venv\Scripts\python.exe scripts\train_lora.py --run dialogue-tone-v1-20261010 --data data/external_sft_tone-v1 --init-adapter runs/expanded-20260929/checkpoint-epoch-1 --profile natural-v1 --prompt-kind character --epochs 1 --max-length 3072 --lr 2e-5 --assistant-logits-only
```

데이터 준비와 학습은 기존 버전 경로를 덮어쓰지 않는다. 새 실행에는 새 버전 이름이 필요하다. 고정 개발 사례16개에서 동일 문맥·seed·생성 설정으로 비교하고, `--analyze-each-model`은 후보 어댑터 자체의 Analyzer를 검사한다. 개선 근거가 부족하거나 명확한 문제가 남은 후보는 Rejected로 기록하며 Baseline 선택을 유지한다. 손실만으로 채택하지 않는다.

## 실제 학습 결과

네 실험 모두 학습40/검증8, 1epoch, optimizer5회, 저장 후 재로드 검증을 완료했다. 아래 손실은 각 실험의 서로 다른 검증셋에서 측정했으므로 모델 순위에 사용하지 않는다. 실험별 설정·커밋·학습 이벤트는 `experiments/results/<ID>-training.json`에 있다.

| ID | 검증 및 재로드 loss | 최대 GPU 할당 GiB | 로컬 체크포인트 |
|---|---:|---:|---|
| tone-v1 | 0.636613 | 20.635 | runs/dialogue-tone-v1-20261010/checkpoint-epoch-1 |
| emotion-v2 | 0.767732 | 20.700 | runs/dialogue-emotion-v2-20261010/checkpoint-epoch-1 |
| mixed-v3 | 0.716425 | 20.706 | runs/dialogue-mixed-v3-20261010/checkpoint-epoch-1 |
| multiturn-v4 | 0.666036 | 20.753 | runs/dialogue-multiturn-v4-20261010/checkpoint-epoch-1 |

실험은 공통 파이프라인 커밋 `fbef3f4`에서 독립 분기했다. 학습 시작 커밋은 순서대로 `95d4397`, `58598af`, `6abb9ac`, `2b32771`이며, 모든 초기 어댑터 SHA256은 동일하다. 저장된 네 어댑터와 Baseline의 SHA256은 서로 다르다. 각 학습 후 Baseline 파일의 해시가 변하지 않았음을 확인했다.

브랜치는 사용자 Fork에 실제 Push했다: [Tone V1](https://github.com/KORemc2/healthmon-capstone/tree/exp/llm/tone-v1), [Emotion V2](https://github.com/KORemc2/healthmon-capstone/tree/exp/llm/emotion-v2), [Mixed V3](https://github.com/KORemc2/healthmon-capstone/tree/exp/llm/mixed-v3), [Multi-turn V4](https://github.com/KORemc2/healthmon-capstone/tree/exp/llm/multiturn-v4). 통합 코드와 모델 목록은 [feat/llm-natural-dialogue](https://github.com/KORemc2/healthmon-capstone/tree/feat/llm-natural-dialogue)에 있다. 현재 원본 PR 생성은 작업 범위에서 제외한다. 앞서 생성한 원본 PR #2는 닫았고 병합되지 않았다.

## 보존과 검증

기존 기준선 태그는 `baseline/llm-local-20261010`, 코드 커밋은 `41e21eb`이다. 로컬 백업 `E:\qwen-llm\backups\natural-baseline-20261010`의 2,992파일/24,303,647,199바이트를 SHA256으로 검증했다. 백업에는 비밀값이 있어 공개하지 않았다. Docker 데몬에 접속하지 못해 Docker 볼륨 백업은 확인하지 못했다. 기존 모델·원본 학습 데이터·운영 DB를 삭제하거나 덮어쓰지 않았다.

복구는 `scripts/model_registry.py --select baseline-expanded`를 실행하고 서버를 재시작한다. 코드 복구·개별 모델·DB 복원 절차는 `docs/NATURAL_DIALOGUE_WORK.md`에 있다. 현재 원본 구현 폴더도 보존돼 있다.

최종 기본 `pytest` 실행에서 467개가 47.11초에 통과했다. 데이터 복원 검사2개와 GPU 잠금 검사1개를 포함한 수치이며, 기존 Node API 검사5개도 통과했다. 외부 원본9파일·변환 split12파일·기존 데이터6파일의 해시를 실제 대조했다. 자동 필터가 만든 실험 내 대화 그룹은 train/validation/test 간 겹치지 않는다. Git 추적 파일의 실제 로컬 비밀값 검사 결과는 0건이었다.

Clone 내부 캐시와 임시 PR 생성 스크립트, 학습 전 실패한 데이터 준비 초안만 정리했다. 원본·Baseline·선택 모델 및 새 실험 체크포인트는 유지했다. 캐시는 이후 테스트 실행으로 다시 생성될 수 있다.

## 고정 개발 사례 비교

각 모델에 같은 사례16개를 전달해 총80응답을 실제 생성했다. 문맥 hash·seed·생성 설정을 통일했고 모델별 실제 어댑터로 분류6건씩 총30건도 생성했다. 원시 응답과 프롬프트는 `experiments/results/dialogue-suite-fixed.json`에 있다. 외부 시험 원문은 이 공개 파일에 포함하지 않았다.

| 모델 | fallback / 16 | 질문 포함 / 16 | 분류 참조와 완전 일치 / 6 |
|---|---:|---:|---:|
| Baseline | 0 | 2 | 4 |
| Tone V1 | 0 | 2 | 4 |
| Emotion V2 | 0 | 4 | 5 |
| Mixed V3 | 0 | 2 | 4 |
| Multi-turn V4 | 0 | 5 | 4 |

위 분류 참조는 에이전트가 작성했고 사람 검증은 하지 않았다. 특히 명시적 제안이 없는 관계 부정 발화의 `acceptance` 참조는 해석이 모호하다. 따라서 일치수를 검증된 정확도로 해석하지 않는다. 실제 필드 이름은 `analysis_confidence`이며, 캐릭터 장기 성격의 `confidence`와 구분한다.

대화에서는 네 후보 모두 개선을 입증하지 못했다. Tone/Mixed는 말다툼의 경위를 모르는 상태에서 사용자 잘못이 아니라고 단정했다. Emotion은 친구의 의도를 추측하고 무시하라고 권했다. Multi-turn은 친구를 탓하고 메일을 오래 쓴 이유를 잠들었기 때문이라고 추측했다. 높은 Affinity 사례의 표현은 어색했고, 상세 설명 요청에서도 설명이 부족했다. 질문 감소나 작은 검증 손실만으로 이를 채택하지 않는다. Baseline에도 같은 종류의 문제가 남아 있어 운영 승인 모델로 취급하지 않는다.

## 별도 시험과 최종 판정

학습에 사용하지 않은 일상2·감정2·멀티턴2·별도 Memory2 사례를 다섯 모델에 전달해 총40응답을 실제 생성하고, 분류2건씩 총10건도 검사했다. 외부 시험의 원시 문장·응답은 `runs/dialogue-heldout-20261010.json`에만 보관한다. 공개 파일 `experiments/results/dialogue-heldout-summary.json`에는 집계값과 데이터 hash만 있다.

| 모델 | 첫 시도 출력 계약 통과 / 8 | fallback / 8 | Memory 근거 ID 및 내용 / 2 | 분류 참조 일치 / 2 |
|---|---:|---:|---:|---:|
| Baseline | 6 | 0 | 2 | 1 |
| Tone V1 | 8 | 0 | 2 | 1 |
| Emotion V2 | 7 | 0 | 2 | 1 |
| Mixed V3 | 8 | 0 | 2 | 1 |
| Multi-turn V4 | 6 | 1 | 2 | 1 |

Memory2사례의 통과만으로 일반 정확성을 입증하지 않는다. 첫 시도에서 실패한 출력은 기존 수정 절차를 거치며, fallback 수도 함께 표시했다. 분류 참조는 위와 같은 미검수 한계가 있다. 사람 평가 점수는 자동으로 작성하지 않았다.

네 학습 후보는 모두 **Rejected**, 선택 모델은 **baseline-expanded**로 유지했다. 새 어댑터를 기본 serving에 적용하지 않았고, Rejected 가중치도 삭제하지 않았다. UI에서는 Rejected를 포함한 비교가 가능하다. 개별 결과와 판정 이유는 `experiments/results/<ID>-evaluation.json`에서 확인한다.

남은 문제는 근거 없는 감정·잘잘못 추정, 높은 Affinity에서의 표현 일관성, 상세 요청에 비해 짧은 응답이다. 소규모·짧은 대화 위주의 데이터 편향과 분석 참조의 모호함도 확인했다. 다음 학습에는 사람 검수와 균형 잡힌 장단문·성격 조건 자료가 필요하다. 이번 결과로 자연스러움 개선이나 운영 적합성을 입증했다고 주장하지 않는다.
