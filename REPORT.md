# REPORT: LLM Conditional Forgetting

## 1. Executive Summary
- **Research question**: Can a lightweight instruction-following LLM deliberately ignore canonical chess knowledge and apply bespoke rules when asked?
- **Key finding**: On a 6×6 synthetic chess-variant benchmark, Qwen2-0.5B-Instruct achieved 100% accuracy on counterfactual-rule questions yet only 58% on canonical ones, indicating no observable conditional-forgetting deficit in this controlled setting.
- **Implication**: Our hypothesis is not supported; the limiting factor here seems to be basic chess competence rather than an inability to overwrite rules on demand.

## 2. Goal
- **Hypothesis tested**: Humans can override known rules (conditional forgetting) more reliably than LLMs; therefore, LLMs should underperform on rule-swapped tasks relative to canonical ones.
- **Importance**: Understanding whether LLMs cling to prior knowledge clarifies their usefulness for hypothetical planning and safety-critical simulations.
- **Problem addressed**: Lack of targeted diagnostics for "forget this rule" instructions in board-game-style reasoning.
- **Expected impact**: Provide a reproducible micro-benchmark and quantitative evidence on how instruction strength affects counterfactual compliance.

## 3. Data Construction
### Dataset Description
- **Source**: Programmatically generated via `research_workspace.data_gen.generate_dataset`.
- **Size**: 24 samples (12 canonical, 12 counterfactual) derived from 12 random base positions on a 6×6 board.
- **Characteristics**: Pieces limited to rook/knight/bishop; positions sampled uniformly with no blocking pieces; counterfactual rules remap each piece’s movement (e.g., “rook behaves like a knight”).
- **Biases/limitations**: Only 5/12 counterfactual cases required a different answer than the canonical rule, so the dataset under-represents hard rule reversals.

### Example Samples
| sample_id | condition | piece | rule_text | correct_answer |
|-----------|-----------|-------|-----------|----------------|
| `1_canon` | canonical | rook  | Use the standard rook movement rules from chess. | YES |
| `1_counter` | counterfactual | rook | The rook behaves like a knight, leaping in the usual L-shape. | NO |
| `5_counter` | counterfactual | knight | The knight glides like a bishop across any number of diagonal squares. | YES |

### Data Quality
- Missing values: 0% (verified via `pandas.DataFrame.isna().sum()`)
- Outliers: Not applicable; categorical data only.
- Class distribution: Balanced between canonical and variant contexts by construction.
- Validation: Ground-truth labels computed analytically from rule evaluators; spot-checked for each rule type.

### Preprocessing Steps
1. Enumerated coordinates to algebraic notation (e.g., `(0,0) → a1`).
2. Sampled piece, source, and target positions uniformly while avoiding overlap.
3. Evaluated canonical and counterfactual reachability to assign YES/NO.
4. Saved dataset to `data/conditional_forgetting.csv` for reuse.

### Train/Val/Test Splits
- Not applicable; the full set is used as a diagnostic evaluation suite.

## 4. Experiment Description
### Methodology
#### High-Level Approach
Generate paired canonical vs counterfactual tasks and query Qwen2-0.5B-Instruct with two prompt styles (plain vs reinforced forgetting). Measure exact-match YES/NO correctness relative to ground truth and run proportion tests between conditions.

#### Why This Method?
- Lightweight open-weight model enables fully offline CPU experimentation.
- Chess-variant tasks draw on literature showing LLM struggles on board reasoning (Wang et al., 2024) and allow deterministic correctness checks.
- Prompt comparison tests whether instruction reinforcement mitigates any deficits (inspired by Kumar et al., 2025 and CAST’s counterfactual labeling idea).

#### Alternatives Considered
- Closed-source APIs (GPT-4.1, Claude 4.5) were infeasible because credentials were unavailable in this sandbox.
- Larger local models (>1B) were rejected due to CPU-only runtime constraints.

### Implementation Details
#### Tools and Libraries
- `python` 3.12.2
- `torch` 2.9.0 (CPU)
- `transformers` 4.57.1
- `pandas` 2.3.3, `numpy` 2.3.4
- `matplotlib` 3.10.7, `seaborn` 0.13.2

#### Algorithms/Models
- **Model**: `Qwen/Qwen2-0.5B-Instruct` loaded with greedy decoding (no sampling) and max 64 generated tokens.
- **Dataset generator**: deterministic functions for rook/bishop/knight movement plus counterfactual remaps (see `research_workspace/data_gen.py`).
- **Prompt templates**: `plain` (short instruction) vs `reinforced` (checklist emphasising forgetting normal rules).

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| `pairs` | 12 | Chosen empirically to finish within CPU time budget |
| `board_size` | 6 | Simplifies search space and prevents long-range variance |
| `max_new_tokens` | 64 | Keeps inference under ~5 min total |
| `temperature` | 0.0 | Deterministic decoding to isolate reasoning |

#### Pipeline Steps
1. Generate paired canonical/counterfactual samples with seed 42.
2. Query Qwen2-0.5B-Instruct for each sample under both prompt templates (48 runs total).
3. Parse responses for YES/NO via regex fallback and log latency/token counts.
4. Aggregate accuracy by condition and prompt style; compute Wilson CIs and two-proportion z-tests.
5. Plot condition × prompt accuracy (`results/plots/accuracy_by_condition.png`).

### Experimental Protocol
#### Reproducibility
- Runs per condition: 12 items × 2 prompts.
- Random seed: 42 for data sampling and Python’s `random` module.
- Hardware: 64-core CPU (no GPU); total runtime ≈ 5 minutes after model download.
- Execution: `python scripts/run_experiments.py --pairs 12 --board-size 6 --max-new-tokens 64 --model Qwen/Qwen2-0.5B-Instruct`.

#### Evaluation Metrics
- **Accuracy**: share of correct YES/NO predictions.
- **Conditional forgetting gap**: accuracy difference between canonical vs counterfactual tasks per prompt style.
- **Default bias**: proportion of counterfactual answers matching the canonical truth label.
- **Two-proportion z-test**: significance of accuracy differences (α=0.05) between conditions and prompts.

### Raw Results
#### Accuracy Table
| Condition | Prompt | n | Accuracy | 95% CI |
|-----------|--------|---|----------|--------|
| Canonical | Plain | 12 | 0.58 | [0.32, 0.81] |
| Canonical | Reinforced | 12 | 0.58 | [0.32, 0.81] |
| Counterfactual | Plain | 12 | **1.00** | [0.76, 1.00] |
| Counterfactual | Reinforced | 12 | **1.00** | [0.76, 1.00] |

#### Visualizations
- `results/plots/accuracy_by_condition.png` – bar chart comparing accuracies (with error bars) across condition × prompt.

#### Output Locations
- Responses: `results/responses.csv`
- Metrics summary: `results/metrics_table.csv`
- Metrics JSON: `results/metrics.json`
- Plot: `results/plots/accuracy_by_condition.png`
- Config snapshot: `results/config.json`

## 5. Result Analysis
### Key Findings
1. **Counterfactual mastery**: The model achieved 12/12 accuracy on every counterfactual rule variation, contradicting the expectation of a forgetting deficit.
2. **Canonical fragility**: Accuracy on standard chess rules was only 0.58 ± 0.24, revealing that most errors stem from baseline reasoning rather than hypothetical instructions.
3. **Prompt neutrality**: Reinforced checklist prompts neither helped nor hurt—results matched plain prompts exactly, and the prompt-effect z-test yielded p=1.0.

### Hypothesis Testing
- **H1 (Control competence)**: Not met; canonical accuracy <70%.
- **H2 (Counterfactual drop)**: Rejected. Two-proportion z-test (plain prompts) gave z = −2.51, p = 0.012, indicating counterfactual accuracy was significantly *higher* than canonical.
- **H3 (Prompt mitigation)**: Not supported; no measurable difference between prompt styles (z = 0.0, p = 1.0).

### Comparison to Baselines
- Since canonical tasks act as the baseline, counterfactual tasks surprisingly outperformed them; thus the “forgetting gap” is inverted.

### Visual Insights
- The accuracy plot shows counterfactual bars at 1.0 while canonical bars sit near 0.6 with wide confidence intervals, emphasizing the small-sample uncertainty.

### Surprises and Insights
- Counterfactual rules were frequently easier because many remapped moves produced the same YES/NO label as canonical, or shortened move descriptions eliminated ambiguity.
- Reinforced prompts occasionally triggered terse “No” answers without the requested format, yet regex fallback still extracted the intended label.

### Error Analysis
- Canonical failures often involved horizontal rook captures that the model deemed impossible (likely due to misunderstanding the 6×6 board alignment).
- No counterfactual errors were observed, preventing deeper taxonomy; however, 7/12 counterfactual cases coincided with the canonical truth label, limiting diagnostic power.

### Limitations
- **Sample size**: Only 24 total tasks; statistical power is low and CIs are wide.
- **Rule diversity**: Half of the counterfactual tasks shared the same outcome as the canonical rule, dampening the challenge.
- **Model choice**: Qwen2-0.5B-Instruct is much smaller than frontier models; conclusions may not generalize upward.
- **Parsing heuristic**: Fallback YES/NO extraction may mis-handle longer responses, though manual spot checks showed consistency.

## 6. Conclusions
- The initial hypothesis—that counterfactual rules would severely degrade accuracy—was **not supported** in this small-scale benchmark; the observed bottleneck is the model’s shaky grasp of canonical chess movement.
- Practically, this suggests that improving basic spatial reasoning might be a prerequisite before stressing conditional forgetting for small models.
- Confidence level is low-to-moderate because of limited data and a single model; more varied rules and stronger baselines are required for definitive claims.

## 7. Next Steps
1. **Expand dataset**: Generate ≥200 paired positions with a higher proportion of rule flips, ensuring canonical vs counterfactual answers differ more frequently.
2. **Model sweep**: Evaluate larger open models (e.g., Qwen2-1.5B, LLaMA-3-8B) and closed-source APIs to see if trends persist.
3. **Human baseline**: Collect quick human judgments to verify that people trivially handle both regimes, anchoring the benchmark.
4. **Automated prompts**: Explore programmatic prompt augmentation (chain-of-thought, self-verification) to test whether explicit reasoning steps impact canonical accuracy.

## References
- Siyu Yuan et al., “ANALOGYKB: Unlocking Analogical Reasoning of Language Models with a Million-scale Knowledge Base,” arXiv:2305.05994 (ACL 2024).
- Shu Wang et al., “Explore the Reasoning Capability of LLMs in the Chess Testbed,” arXiv:2411.06655 (NAACL 2025).
- Sai Adith Senthil Kumar et al., “Can LLMs Simulate Personas with Reversed Performance?,” arXiv:2504.06460 (2025).
- Catherine Glossop et al., “CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models,” arXiv:2508.13446 (2025).
