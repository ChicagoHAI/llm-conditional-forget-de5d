# Planning

## Research Question
Can current instruction-following LLMs correctly apply newly introduced, counterfactual chess-like rules when they explicitly contradict their prior knowledge, or do they default to canonical rules despite conditional forgetting instructions?

## Background and Motivation
Humans routinely override prior knowledge in hypothetical reasoning (e.g., "imagine knights move like bishops"). Recent work on analogical datasets (Yuan et al., 2024) and counterfactual instruction following (Kumar et al., 2025) shows that LLMs still mis-handle such scenarios, yet systematic diagnostics around rule overriding remain sparse. Chess reasoning benchmarks (Wang et al., 2024) highlight that board games expose planning blind spots. A controlled, synthetic setup will let us isolate whether models can truly “forget” learned rules when explicitly told to.

## Hypothesis Decomposition
1. **H1 (Control Competence)**: LLM achieves ≥70% accuracy on canonical-rule questions (baseline). Independent variable: task uses standard chess rules. Dependent variable: accuracy.
2. **H2 (Conditional Forgetting Failure)**: Accuracy drops significantly (≥20 percentage points) on counterfactual-rule questions even when instructions are explicit.
3. **H3 (Prompt Mitigation)**: Adding structured reminders (system + step-by-step admonitions) reduces the failure gap relative to a plain instruction prompt.

## Proposed Methodology

### Approach
- Generate paired datasets of chess position questions with programmatic ground truth on a reduced board (6×6 to simplify computation).
- Create two task regimes: (a) **Canonical** (standard piece moves); (b) **Counterfactual** (pieces reassigned to other moves or limited ranges).
- Use a lightweight open-weight chat model (TinyLlama-1.1B-Chat) to simulate responses under two prompt templates (plain vs reinforced conditional forgetting) due to CPU-only constraints.
- Measure exact-match correctness (YES/NO) and compute McNemar/chi-square tests between regimes.

### Experimental Steps
1. **Dataset generation**: Implement script to sample board states, assign piece types, compute reachability, and output JSON/CSV.
2. **Prompt design**: Develop two prompt templates: baseline instruction and reinforced “forget old rules” instruction with checklist.
3. **Inference runs**: Query the model for each scenario under both templates, capture responses, and parse structured outputs.
4. **Metrics computation**: Compare outputs to ground truth, compute accuracy, confusion matrices, and statistical tests.
5. **Analysis & visualization**: Plot accuracy by regime, rule type, and prompt template; examine error categories qualitatively.

### Baselines
- **Canonical-rule accuracy** acts as sanity baseline ensuring the model understands the task when rules match prior knowledge.
- **Plain prompt** vs **reinforced prompt** isolates whether better instruction engineering mitigates failures.

### Evaluation Metrics
- **Accuracy**: proportion of questions answered correctly.
- **Conditional forgetting gap**: accuracy difference between canonical and counterfactual tasks.
- **McNemar test** on paired items to judge significance when the same positions are used with swapped rules.
- **Response latency / token count** (optional) for noting if counterfactual reasoning induces longer outputs.

### Statistical Analysis Plan
- Use Wilson 95% confidence intervals for accuracy per condition.
- Paired McNemar test between canonical vs counterfactual within each prompt template (H2) and between prompt templates within counterfactual regime (H3).
- Report p-values and effect sizes (difference in proportions) with Cohen’s h.

## Expected Outcomes
- Support for hypothesis if counterfactual accuracy is substantially lower and reinforced prompts only partially recover it.
- Refutation if counterfactual accuracy matches canonical accuracy without large gaps.

## Timeline and Milestones
- Phase 0–1 (already underway): 20 min.
- Data & tooling setup: 25 min.
- Model inference runs: 20 min (batching small dataset of ~60 items × 2 templates).
- Analysis & visualization: 25 min.
- Reporting & validation: 30 min (buffer for reruns / documentation).

## Potential Challenges
- TinyLlama may already struggle on canonical chess tasks; mitigate by simplifying board and rules.
- CPU inference may be slow; keep dataset modest and cache responses.
- Parsing free-form outputs could be noisy; enforce constrained answer format (“FINAL ANSWER: YES/NO”).

## Success Criteria
- Reproducible dataset + code with deterministic seeds.
- Completed experiments with logged outputs in `results/`.
- REPORT.md detailing statistically supported conclusions about conditional forgetting behavior.
