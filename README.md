# LLM Conditional Forgetting Workspace

## Overview
This workspace investigates whether large language models can temporarily discard known chess rules and obey newly introduced counterfactual variants. A synthetic 6×6 chess-variant dataset was generated and Qwen2-0.5B-Instruct was evaluated with two prompt templates (plain vs reinforced reminders to forget old rules).

## Key Findings
- Counterfactual tasks were solved perfectly (12/12), while canonical tasks reached only 58% accuracy, so no forgetting deficit surfaced.
- Prompt reinforcement yielded identical scores to the plain template, implying instruction strength was not the bottleneck.
- Most canonical errors were simple rook or bishop moves, indicating the small model’s basic chess knowledge is the limiting factor.

## Reproducing the Experiments
1. Create/activate the uv environment (already provided in this session):
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```
2. Run the full pipeline (re-generates data, reruns inference, overwrites results):
   ```bash
   python scripts/run_experiments.py --pairs 12 --board-size 6 --max-new-tokens 64 --model Qwen/Qwen2-0.5B-Instruct
   ```
3. Inspect outputs in `results/` and plots in `results/plots/`.

## File Structure
- `resources.md` – literature survey and dataset rationale.
- `planning.md` – experimental design and hypotheses.
- `research_workspace/` – reusable modules (data generation, prompting, evaluation helpers).
- `scripts/run_experiments.py` – end-to-end pipeline (data → inference → metrics → visualization).
- `data/conditional_forgetting.csv` – generated benchmark instances.
- `results/` – raw model responses, metrics, config snapshot, and plots.
- `REPORT.md` – detailed study write-up with analysis and references.

## More Information
See `REPORT.md` for the complete methodology, statistical analysis, visualizations, and next-step recommendations.
