# Resource Survey

## Literature & Benchmarks Consulted

1. **Yuan et al. (2024) – ANALOGYKB: Unlocking Analogical Reasoning of Language Models with a Million-scale Knowledge Base** ([arXiv:2305.05994](http://arxiv.org/abs/2305.05994))
   - Establishes that LLMs benefit from structured analogy resources yet still lag behind humans on compositional reasoning. Highlights the need for diagnostics that isolate the ability to override prior associations when rules change.

2. **Wang et al. (2024) – Explore the Reasoning Capability of LLMs in the Chess Testbed** ([arXiv:2411.06655](http://arxiv.org/abs/2411.06655))
   - Provides evidence that even instruction-tuned LLMs falter on chess reasoning requiring long chains of tactic + strategy. Suggests chess is a good substrate for evaluating conditional rule use.

3. **Kumar et al. (2025) – Can LLMs Simulate Personas with Reversed Performance?** ([arXiv:2504.06460](http://arxiv.org/abs/2504.06460))
   - Introduces a benchmark for counterfactual instruction following and finds modern LLMs fail to obey reversed personas. Motivates focusing on "conditional forgetting" scenarios where models must discard default knowledge.

4. **Glossop et al. (2025) – CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models** ([arXiv:2508.13446](http://arxiv.org/abs/2508.13446))
   - Demonstrates that explicitly generating counterfactual labels improves adherence to hypothetical instructions in embodied agents. Informs our plan to create synthetic counterfactual rule data with tight formatting to encourage compliance.

## Workspace Audit

- Checked directories (`artifacts/`, `logs/`, `notebooks/`, `results/`); currently empty and contain no provided datasets or notes.
- `.idea-explorer/` exists but per instructions will remain unused.

## Dataset Decision

- No ready-made dataset targeting "conditional forgetting" in chess-like settings was found in the workspace or literature review.
- Inspired by prior work, I will generate a **synthetic counterfactual chess-variant dataset** featuring:
  - Standard-rule control items (baseline performance check).
  - Counterfactual-rule items (e.g., "knights move like bishops"), forcing models to ignore canonical chess knowledge.
  - Programmatically computed ground truth for reproducibility.
- Synthetic generation is justified because it gives fine-grained control over rule permutations, ensuring the evaluation directly targets the hypothesis.

## Tooling and Model Choice Rationale

- Will run experiments on CPU using a lightweight open-weight chat model (`TinyLlama-1.1B-Chat`) via `transformers`. Larger APIs (GPT-4.1, Claude 4.5) would be ideal but require external credentials that are unavailable in this sandbox; limitations will be documented.
- Python stack will include `numpy`, `pandas`, `scipy`, `matplotlib`, and `transformers` for data handling, statistics, visualizations, and inference.

## Next Steps

1. Formalize research plan (planning.md) with metrics, baselines, and analysis strategy.
2. Implement synthetic data generator + evaluation harness.
3. Run LLM experiments under baseline vs conditional-forgetting prompts, collect metrics, and document findings in REPORT.md.
