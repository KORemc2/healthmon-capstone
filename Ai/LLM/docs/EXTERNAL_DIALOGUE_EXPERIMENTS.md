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

데이터 준비와 학습은 기존 버전 경로를 덮어쓰지 않는다. 새 실행에는 새 버전 이름이 필요하다. 고정 개발 사례16개에서 동일 문맥·seed·생성 설정으로 비교하고, `--analyze-each-model`은 후보 어댑터 자체의 Analyzer를 검사한다. 결과는 사람 검토 전까지 Candidate로 보관하며 Baseline 선택을 유지한다. 손실만으로 채택하지 않는다.
