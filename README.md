---
license: other
license_name: mixed-source-research-artifacts
license_link: LICENSES/README.md
language:
- en
tags:
- llm-evaluation
- inference
- reproducibility
- humaneval
- math-500
- gpqa
- model:sarvamai/sarvam-30b
- model:Qwen/Qwen3.8-27B
pretty_name: General harness repair — Sarvam and Qwen evidence
configs:
- config_name: sarvam_humaneval_plus
  data_files:
  - split: test
    path: answers/sarvam/paired/humaneval_plus.jsonl
- config_name: sarvam_math500
  data_files:
  - split: test
    path: answers/sarvam/paired/math500.jsonl
- config_name: sarvam_gpqa_diamond
  data_files:
  - split: test
    path: answers/sarvam/paired/gpqa_diamond.jsonl
- config_name: qwen_humaneval_plus
  data_files:
  - split: test
    path: answers/qwen/paired/humaneval_plus.jsonl
- config_name: qwen_math500
  data_files:
  - split: test
    path: answers/qwen/paired/math500.jsonl
- config_name: qwen_gpqa_diamond
  data_files:
  - split: test
    path: answers/qwen/paired/gpqa_diamond.jsonl
---

# A general harness repair approach to AI Model improvement

**Jay Desai — Gujarat University.**

This repository accompanies the [final paper](paper/A_General_Harness_Approach_Final.pdf). It contains the executed harness source, complete permitted baseline/repaired responses, grading records, review provenance and reproducibility identifiers for Sarvam 30B and Qwen3.8-27B. Original upstream weights are unchanged. No training, adapters or weight edits were performed.

The policy was developed on Sarvam, frozen for a fresh full Sarvam comparison, and transferred before Qwen's scored evaluation. These public benchmarks were used in development; checkpoint transfer is not an unseen-task holdout. The two-arm comparison measures the whole executed bundle, including a generic mathematics instruction change. It does not isolate the closing delimiter's causal effect.

## Primary automatic results

| Model | Test | Original baseline | Frozen repaired | Net | Gains / losses |
|---|---|---:|---:|---:|---:|
| Sarvam 30B | HumanEval+ | 95/164 (57.93%) | 146/164 (89.02%) | +51 | 53 / 2 |
| Sarvam 30B | MATH500 | 372/500 (74.40%) | 435/500 (87.00%) | +63 | 84 / 21 |
| Sarvam 30B | GPQA Diamond | 50/198 (25.25%) | 109/198 (55.05%) | +59 | 62 / 3 |
| Qwen3.8-27B | HumanEval+ | 150/164 (91.46%) | 153/164 (93.29%) | +3 | 6 / 3 |
| Qwen3.8-27B | MATH500 | 476/500 (95.20%) | 487/500 (97.40%) | +11 | 19 / 8 |
| Qwen3.8-27B | GPQA Diamond | 135/198 (68.18%) | 173/198 (87.37%) | +38 | 38 / 0 |

Primary definitions: HumanEval original **and** expanded execution success on the logical raw view; frozen MATH500 v3 equivalence scoring; pinned simple-evals first-match raw GPQA extraction. See `results/` for every named secondary view. Secondary final-channel scores are not substituted into this table.

## Separate Codex review

| Model | Review measure | Baseline | Repaired |
|---|---|---:|---:|
| Sarvam 30B | Code: no static defect found | 91/164 | 143/164 |
| Sarvam 30B | Mathematics: resolved final value | 454/500 | 460/500 |
| Sarvam 30B | GPQA: reviewed selected choice matches key | 69/198 | 114/198 |
| Qwen3.8-27B | Code: no static defect found | 149/164 | 149/164 |
| Qwen3.8-27B | Mathematics: resolved final value | 494/500 | 497/500 |
| Qwen3.8-27B | GPQA: reviewed selected choice matches key | 151/198 | 173/198 |

These are case-by-case **Codex assessments, not independent human grading**. Code review is not execution pass@1 or a correctness proof. Mathematics assesses resolved final values, not full proofs. GPQA review extracts the selected option and compares it with the upstream key; it is not independent scientific adjudication. All original labels, unresolved cases, regressions, provenance and declared sensitivities are retained. The separate Sarvam numerical-stress sensitivity is **89 → 139**, not a replacement for either original static labels or automatic scores. See the paper for inconsistent public specifications and review limitations.

## Conditions

- All 164 code, 500 mathematics and 198 science problems per arm: **862 per arm, 1,724 per model, 3,448 total**.
- One resulting completion per item and arm. No best-of-k, voting, correctness-feedback retries or answer substitution.
- Code output ceiling **4,096 tokens**; math/science **8,192**. Repaired reserve **1,024**, including the two inserted delimiter tokens. Context ceiling **12,288**.
- Greedy decoding: temperature 0, top-p 1, seed 20261108. At most 48 sequences, 4,096 batched tokens, eager mode, no prefix cache. Qwen retains native `xhigh` effort in both arms.
- Same graders across each comparison. Code and GPQA rendered prompts match; mathematics has a disclosed uniform instruction change.
- A100-SXM4-80GB. Model-specific runtimes are pinned in the protocols and environment records. Original BF16 weights; Sarvam retains original FP32 routers. The unused Qwen vision encoder is skipped.
- No references, hidden tests, canonical programs or scoring feedback were supplied to the GPU solver. The inserted policy text is only `</think>` plus a newline. The policy has no question/task-ID branch.

## What is released

| Location | Contents |
|---|---|
| `paper/` | Author-edited final PDF and the explicitly authorized GPQA disclosure correction |
| `source/sarvam/`, `source/qwen/` | Exact frozen executed workers, budget policy, channel handling, native adapter and evaluator sources |
| `protocols/` | Derived public protocol copies; original protocol hashes retained; personal funding statements removed |
| `answers/*/paired/` | Every baseline/repaired code/math completion, delivered final, evaluated program where relevant, verdict and review label; GPQA metadata only |
| `answers/*/token-records/` | Lossless gzip copies of all primary code/math generation records, including sampled/forced token IDs |
| `reviews/`, `evaluation/` | Code/math review ledgers, actual sanitized programs, both execution suites and raw/final mathematics verdicts |
| `inputs/` | Code/math question-only projections, the exact full question order, prompt templates and authorized local GPQA reconstruction recipe |
| `references/math500/` | Attributed MATH references for isolated CPU scoring only |
| `results/`, `provenance/` | Aggregate results, recomputed counts, original source hashes, environment metadata, freeze identities and sensitivities |
| `historical/` | Permitted historical Sarvam development answers, verdicts, shared-prefix/sham evidence and review records, separate from final scores |
| `exploratory/` | Subsequent unsuccessful Qwen-specific probes and a user-stopped partial duplicate, clearly separate from the paper's completed study |
| `archive/experiment-tools/` | Original orchestration/review tools for historical audit; these depend on the original workspace layout |

**GPQA answers are withheld in accordance with the benchmark guidelines.** No GPQA question text, options, selected letters, reference keys, reasoning traces, response token IDs or example-bearing screenshots are published. All 792 primary GPQA response records are represented by per-ID correctness/resolution flags, counts and hashes. Full public response text covers the other **2,656 primary responses**. Use the authorized upstream dataset locally to reproduce GPQA.

The original model checkpoints are linked, not duplicated or represented as newly trained models:

- [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b), revision `071ae95e933605ca1104a6b4524a36a98488efa4`.
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Inspect and reproduce

```sh
python tools/audit_release.py
python tools/recompute_scores.py
```

These commands validate retained files and recompute counts from saved verdicts. They do **not** run model-generated code, allocate a GPU, or establish fresh independent accuracy. See [reproduction instructions](docs/REPRODUCTION.md) for an actual rerun, runtime isolation, exact input reconstruction and retained-source limitations. See [limitations and study chronology](docs/STUDY_NOTES.md) before quoting results.

## Open release and citation

- GitHub: https://github.com/jaydesaigu-arch/general-harness-repair
- Hugging Face evidence dataset: https://huggingface.co/datasets/DesaiJayM/general-harness-repair
- arXiv: submission status is recorded separately; no acceptance or public identifier is claimed before arXiv issues one.

A [complete artifact download](https://github.com/jaydesaigu-arch/general-harness-repair/releases/tag/v1.0.0) is provided with the release. Use `CITATION.cff`. Original project code, documentation and new annotations are Apache-2.0. The paper is distributed here with the author's publication authorization; its additional reuse license awaits the author's arXiv license choice. Third-party model/code/data licenses and attribution remain applicable; see [license scope](LICENSES/README.md) and [notices](NOTICE.md).
