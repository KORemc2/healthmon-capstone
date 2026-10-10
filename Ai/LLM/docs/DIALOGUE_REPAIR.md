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

Current stage: semantic triage and final dataset freeze before full training. Completed training/evaluation results will be appended only after they actually finish.
