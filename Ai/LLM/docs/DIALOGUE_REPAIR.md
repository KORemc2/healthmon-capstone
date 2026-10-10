# Dialogue quality repair — 2026-10-10

Work is confined to the clone of `KORemc2/healthmon-capstone`. Original repositories receive no writes. Baseline data, weights and earlier results remain recoverable.

## Reproduced failures

The four previous runs used 40 training rows, one epoch and only five optimizer updates each, initialized independently from the same expanded baseline. Calling these successful quality improvements was not justified. All four remain Rejected.

CPU inspection found different parameters in all 496 adapter tensors, with relative adapter parameter L2 deltas of approximately 1.05–1.17% from baseline. They are not accidentally identical checkpoint files. Their training speech averaged only 17–27 characters. The preparer imposed a 100-character answer ceiling and sampled from pools capped at 128 rows per type/split.

Actual ablations are stored locally under `runs/dialogue-repair-20261010/diagnostic.json`. Tone reproduces `할 수 없어.` on the user's exercise reluctance example. This exact target is absent from its training data; inference reports a real first-attempt generation, not a fallback. We cannot attribute the phrase to one particular training row.

Both base Qwen and adapted models can redirect boredom to the food-record candidate under the old large prompt. Both can describe suggested movements as completed. Removing unrelated candidate exposure and using native conversation turns corrected the instruction-following example in the prompt-only ablation. The prompt changes are a combined intervention; this does not isolate every individual cause.

## Implementation

`natural-v2` is an opt-in Character prompt. Baseline and natural-v1 remain available for honest comparison. Recent user/assistant turns use native roles; the actual last request is a user message. A compact projection retains Memory IDs, active quest goals/progress, verified activity facts, physical state and growth state. General suggestions use `action=none`; explicit quest requests retain authorized catalog candidates. Numeric growth calculations are unchanged.

Training supports NF4 double quantization with BF16 compute, language-only LoRA, assistant-selected vocabulary logits, padded batches, warmup/decay and optional higher loss weight on the speech string. Selected-logit CE and gradients are checked against full causal CE for single examples, padded batches and weighted speech.

The first actual NF4 sanity run completed backward passes, two optimizer updates, adapter save/reload and identical validation loss. Peak allocated memory was 13.04 GiB. Earlier long-context BF16 experiments peaked near 20.7 GiB; context lengths differ, so this is not an isolated quantization benchmark. Dependencies are isolated in this clone's `runtime/qlora-deps`; the original Python environment is unchanged.

## Data preparation and limitations

An initial expansion produced 1,375 rows, but manual spot checks found unsuitable targets: encouragement instead of interview tips, unsolicited gym advice, unsupported assumptions and awkward translations. It is retained as a discarded preparation/sanity version, not accepted as proof of quality.

The revised pipeline also checks [Smol-KoreanTalk](https://huggingface.co/datasets/lemon-mint/smol-koreantalk), a Korean translation of Smol-SmolTalk declaring Apache-2.0. Source revision and downloaded file SHA256 values are pinned in `experiments/dialogue-repair-source-full.json`. Role-playing, system overrides, technical and high-stakes topics are filtered. Some long answers become explicitly documented short-answer tasks using complete sentence/list-item excerpts; these are not represented as complete original answers or silent tokenizer truncation.

Local base Qwen performs A/B semantic triage. Its scores are uncalibrated heuristics, not human labels, gold quality judgments or accuracy. Final selection additionally applies text/contract checks, known bad-example exclusions, whole-conversation split checks and substantial exact utterance overlap checks. Semantic paraphrase leakage and unreviewed errors remain possible. Authored task examples and controlled personality expression pairs are labeled as agent-authored, without human approval.

Restricted external source text, derived data and adapters stay local and ignored. The existing CC-BY-NC material continues to limit mixed experiments to noncommercial local research. No paid API calls are used.

## Evaluation gate

Training loss alone cannot select a replacement. Compare actual multi-turn responses against preserved baseline and the same-prompt baseline; retain raw outputs, adapter identities, fixed generation settings and factual/contract failures. Memory, quest IDs, Analyzer outputs and growth/API compatibility require separate checks. No production promotion follows automatically.

Official implementation references: [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

## Frozen full training run

`dialogue_repair_v8` contains 851 training, 40 validation and 44 held-out rows. Training has 739 conversation groups and 358 rows with prior turns. Speech averages 82.3 characters; the maximum is 423. This remains a development-sized dataset, not a large-scale Korean language training corpus. Final source hashes and all selection limitations are recorded in `experiments/repair-v5.json`.

The final configuration passed a separate eight-row sanity run using the longest training examples, batch size 2, rank 16 and speech loss weight 3. It saved and reloaded the adapter with identical validation loss; peak allocated memory was 17.62 GiB. The full run starts independently from the clean pinned base, uses NF4 double quantization, and schedules two epochs / 426 optimizer updates. The earlier baseline and all four rejected adapters remain intact.

Full training completed on 2026-10-10 with 426 optimizer updates and both epoch checkpoints preserved. Validation loss was 1.2951634 after epoch 1 and 1.2942645 after epoch 2. Epoch 2 was selected by the lower validation loss and reloaded with the same loss. The difference is small and does not establish dialogue quality. Peak allocated GPU memory was 17.56 GiB. The training source commit is `2d80be40717f427f4a7322a578d42028fc6619c6`; the selected adapter SHA256 is recorded in `experiments/repair-v5.json`.

## Completed comparison and remaining failures

Three configurations were evaluated: preserved baseline, preserved baseline with `natural-v2`, and the new LoRA with `natural-v2`. There are 48 fixed Character generations, 63 personality/multi-turn generations, 36 local held-out generations and 18 additional-seed generations (165 Character generations total), plus 18 Analyzer checks. First-turn comparisons share context and seed; follow-ups use each model's own preceding answers. These are development examples, not an independent benchmark or human ratings.

| Check | Baseline | Baseline + new prompt | New LoRA + new prompt |
| --- | ---: | ---: | ---: |
| Fixed Character fallback responses / 16 | 0 | 0 | 1 |
| Conversation/axis fallback responses / 21 | 0 | 0 | 0 |
| Local held-out fallback responses / 12 | 0 | 0 | 2 |
| Extra-seed fallback responses / 6 | 0 | 0 | 0 |
| Analyzer exact matches to authored expectations / 6 | 4 | 4 | 3 |

The Analyzer expectations are agent-authored; some acceptance/motivation labels are debatable. The match counts are not validated accuracy. The new Character-only adapter also changes Analyzer behavior: for `스쿼트 할 생각에 신나`, motivation changed from the baseline's `high` to `none`. It must not replace the Analyzer without further work.

In the exercise recommendation conversation, baseline describes unperformed suggestions as completed: `1. 2분 스트레칭을 했어. 2. 집에서도 가벼운 어깨 동작으로 시작했어.` The new adapter instead lists actionable steps: `1. 책상 앞에 앉기 / 2. 어깨와 발목 살짝 움직이기 / 3. 통증 있으면 멈추기`. It also respects the tested request to stop exercise talk. These are specific improvements in examples, not proof of overall superiority.

Remaining problems include weak personality separation (Activity 20 and 80 produce the identical new-adapter response), repetitive cup-related suggestions, shallow empathy (`왜 그런 일이 일어났어?`), and inconsistent recommendation behavior across seeds. On seed 20261012 it answers `운동 뭐 하지?` with `오늘은 편하게 쉬어도 돼.` Three fixed seeds do not establish a general refusal rate, even though `할 수 없어.` did not recur in the 27 conversation/axis/extra-seed responses for the new adapter.

Overall improvement is **not established**. Both new configurations remain available as Candidates for inspection; the selected baseline is unchanged, compatibility has not passed, and there is no automatic promotion, human approval or upstream publication.

Actual synthetic outputs are retained in `experiments/results/repair-v5-fixed.json`, `repair-v5-conversations.json` and `repair-v5-seed-checks.json`. `repair-v5-heldout-summary.json` contains only aggregate counts and hashes; restricted held-out source texts remain local. The local comparison UI is restored at `http://127.0.0.1:8002/`. Code, configuration, provenance and permitted evaluation evidence are published only to the owner's Fork; model weights and restricted training data remain local.
