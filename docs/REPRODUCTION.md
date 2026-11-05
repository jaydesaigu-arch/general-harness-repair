# Reproduction and retained-source boundaries

## Inspect the released record without paid compute

Run `python tools/audit_release.py` and `python tools/recompute_scores.py` from the repository root. Python 3.10+ and its standard library suffice. This verifies source identities, lossless saved generation records, budgets, complete ID coverage and the six primary score rows. It does not execute generated code or independently validate the frozen graders' decisions.

`answers/<model>/paired/*.jsonl` has one record per problem and one object per arm. `raw_completion` is the logical complete response used by the named primary scorer. `delivered_final` is the parsed final channel. Qwen's native input-opened `<think>` marker is reconstructed in the logical view; `raw_generated_text` and full token composition are retained in the compressed token records. The input marker is not counted as a newly sampled token. The original native end markers are trimmed from retained token totals; configured sampling limits still bound generation.

The code output in `evaluated_program` is the exact sanitized program evaluated by the frozen execution grader. Both original and expanded suites must pass. Review labels describe static judgments, not execution success. Full code/math review records are under `reviews/`; GPQA review labels containing selected letters remain private.

## Reconstruct the exact question-only solver input

1. Obtain `gpqa_diamond.csv` through an authorized upstream source and comply with its access conditions. Do not post it or its answers to this repository.
2. In a private local directory run:

   ```sh
   python tools/prepare_inputs.py --gpqa-csv /private/gpqa_diamond.csv --output /private/solver/questions.json
   ```

3. The tool verifies the CSV SHA-256, duplicates the pinned `Random(0)` option ordering and uses the released code/math projections plus exact 862-ID order. It must reproduce question-projection SHA-256 `732f96bca7bafc621e65092e3ea343a6b20fd3cf817308aefa1a5e36a6039bba`. It writes question/specification data and offered options, **no answer key**.
4. Optional `--gold-output` creates a separate CPU-only GPQA reference file. Keep it outside the GPU solver mount. Do not publish it.

The same question-only projection is used for both models. Original source schemas contain exactly `id`, `benchmark`, `prompt_before`, `prompt_after`, `output_budget`. The shared prompt templates are disclosed in `inputs/prompt-templates.json`. Code and science prompts match across arms; mathematics differs uniformly as disclosed.

## Actually rerun a model

The exact executed workers and bootstraps are in `source/<model>/`. They are retained research source, not a newly tested turnkey inference library. Their expected filesystem roots are:

- Sarvam: `/workspace/sarvam30b`.
- Qwen: `/workspace/qwen38-27b`.

Stage the corresponding source files, a copy of `protocols/<model>.json` named `protocol.json`, and the privately reconstructed `questions.json` at that root. Supply `limits.json` with `deadline_unix`, a real Unix-time bound. Download the **pinned original checkpoint**, using the original bootstrap to retain tokenizer/config/runtime identity. Original checkpoint revisions, source digests, runtime settings and environment records are included. No weights are redistributed.

Use distinct containers for generation and evaluation. The GPU container must contain no reference keys, canonical functions, hidden tests, grading verdicts or prior responses. The Qwen worker first runs two synthetic timing batches and waits for a `GO.json` marker; review actual runtime/memory/cost before permitting scored generation. Synthetic timing requests are not benchmark attempts. The original job controllers document that decision, artifact harvest and shutdown. They depend on the original workspace and Hugging Face Jobs API; importing/running them can allocate **billable compute**. Do not run archived controllers just to inspect this release.

Baseline runs first and repaired second. The baseline samples the full ceiling naturally; the repaired branch samples at most B−1,024, closes an unfinished leading native reasoning span where required, and continues using the remaining allowance. Forced tokens count against the existing ceiling. Do not add retries, select outputs by correctness, change prompts between tasks, or substitute post hoc fixes for generated programs.

## Evaluate separately

The frozen evaluator sources are in each model's `evaluator/` directory. Their unchanged original container layout is `/workspace/he164` for code, `/workspace/math500` for mathematics and `/workspace/sarvam30b` for shared orchestration, including in the Qwen experiment. `archive/experiment-tools/<model>/cpu_bootstrap.py` qualifies and stages dependencies; `score_cpu.py` records raw and final views with explicit names. Math-Verify 0.9.0, EvalPlus 0.3.1 and HumanEvalPlus-v0.1.10 were used. Preserve the qualified fixes; do not compare a patched repaired grader with an unpatched baseline grader.

MATH references under `references/math500/` are for the **isolated CPU evaluator only**; stage them under the original expected names `private-gold.json` and `private-reference-solutions.json`. Obtain HumanEval/EvalPlus data through its upstream loader on CPU. Obtain GPQA keys locally and apply `grade_gpqa.py` only after independent choice extraction is frozen. All released GPQA per-ID flags keep the upstream denominator of198.

Model-generated programs are untrusted: execute them only in an isolated evaluator with bounded time/resources and no credentials, personal data or network access needed by the generated programs. The audit commands in this repository never execute them.

## Review and variance

The released review ledgers are historical Codex assessments. Exact same-question, same-artifact judgments can be reused with provenance; unmatched cases need fresh inspection. They are not an independent human panel. Fresh independent reproduction should retain every automatic pass and failure for review and should document reviewer identity, blinding, uncertainty and corrections.

One greedy batched run per arm is not a variance estimate. In the original Qwen full pair, none of the forced cases shared an identical sampled prefix with baseline up to the forced transition. The paper therefore reports an executed-bundle comparison, not a clean delimiter-only causal effect. Repeat seeds/order or matched saved-prefix controls for stronger causal claims. Same known tasks are not an unseen-task holdout.
