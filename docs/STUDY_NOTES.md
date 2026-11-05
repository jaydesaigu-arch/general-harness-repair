# Chronology, sensitivities and limits

The manuscript's primary comparison is the full frozen Sarvam comparison and the transfer of that same general policy to Qwen **before Qwen scored inference**. Both checkpoints have862 baseline and862 repaired responses. All regressions remain in the paired files. Those completed results are not replaced by later exploratory experiments.

Sarvam development was benchmark-informed. The shared mathematics instruction was selected during development. This limits task generalization even though Qwen transfer reduces checkpoint-specific tuning. The intervention also includes generation budgeting, native-channel handling, a declared mathematics instruction change and common evaluator fixes. Equal output ceilings do not imply equal compute because continuation replays the prefix.

Code automatic results are finite-suite success, not universal program correctness. Numerical stress witnesses change original Sarvam static no-defect counts from91→143 to89→139; the automatic95→146 remains unchanged. Public specification conventions were not fully harmonized across the two models' historical static reviews. Mathematics alternate-convention sensitivities add two points to each Sarvam arm and one to each Qwen arm, changing neither net reviewed gain. Frozen primary conventions remain in the main tables.

Codex review can miss errors and is not independent human validation. Selected-choice GPQA review is not adjudication of scientific explanations. Mathematical value agreement is not proof verification. No activation, gradient, norm or singular-value localization, new training or learned-weight repair was performed.

## Subsequent unsuccessful Qwen-specific refinements

These experiments occurred **after** the manuscript's completed Qwen transfer comparison and are segregated under `exploratory/`.

A deterministic failure-enriched sample used24 code,32 mathematics and40 GPQA cases, including passing controls. Fresh legacy general-harness automatic counts were16/24,26/32,28/40. A budget-based native-effort/canonical-delimiter candidate produced18/24,22/32,26/40 and failed the selection rule. A delimiter-only candidate with original `xhigh` effort produced15/24,26/32,28/40 and also failed. In the latter probe, all55 forced cases had identical sampled prefixes to the saved legacy comparator. Both outcomes are retained; no new Qwen-specific improvement is claimed.

Pilot graders required full-dataset inputs. Unselected rows were padded with identical saved outputs solely for unchanged grader input validation. **Only the selected24/32/40 cases count** in development results; padded rows are not new generations. The first probe generated192 completions, and the second generated96, reusing the declared saved96-case legacy comparator. Selection on this failure-enriched known-task sample is exploratory, not an estimate of full-benchmark accuracy.

A further duplicate full replication was started unnecessarily and stopped at the user's request after96 baseline responses had been saved (10 code,59 math,27 GPQA). It has no full-suite repaired arm and supplies **no new full benchmark score**. Its retained code/math partial outputs, shutdown verification and cost estimate are separately identified. It is excluded from all primary denominators. No compute is left running for this release.

The original full-pair conservative elapsed allocation estimates were$4.428 for Sarvam and$18.469 for Qwen (including recorded startup attempts for Qwen). Subsequent unsuccessful Qwen development cost about$3.5851, and the interrupted duplicate about$1.5185. These are attributed historical estimates including ports, not invoices or current balances. They are not training costs or a controlled cross-architecture efficiency comparison.

## Publication correction

The human author's final PDF is preserved except for the specifically authorized page10 disclosure paragraph. It now explicitly states that GPQA answers and reasoning traces are not released, consistent with the benchmark guidelines and actual repository contents. All18 pages were checked at pixel level outside the changed paragraph; no other visible content changed. Original and corrected SHA-256 digests are in `paper/publication-correction.json`.
