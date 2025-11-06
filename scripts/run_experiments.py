import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from research_workspace.data_gen import generate_dataset, set_seed
from research_workspace.evaluation import (
    compute_default_bias,
    extract_answer,
    proportions_ztest,
    summarize_by_group,
)
from research_workspace.prompting import build_prompt


def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    return tokenizer, model


def run_inference(df: pd.DataFrame, tokenizer, model, prompt_styles, max_new_tokens: int = 96):
    rows = []
    for _, sample in df.iterrows():
        for style in prompt_styles:
            prompt = build_prompt(sample, style)
            messages = [
                {"role": "system", "content": "You are a meticulous chess analyst who always follows hypothetical rules exactly."},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            latency = time.perf_counter() - start
            generated = outputs[0][input_ids.shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed = extract_answer(text)
            is_correct = parsed == sample["correct_answer"]
            rows.append(
                {
                    **sample.to_dict(),
                    "prompt_style": style,
                    "model_output": text,
                    "parsed_answer": parsed,
                    "is_correct": bool(is_correct),
                    "latency_sec": latency,
                    "output_tokens": int(len(generated)),
                }
            )
    return pd.DataFrame(rows)


def build_plot(summary_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=summary_df,
        x="condition",
        y="accuracy",
        hue="prompt_style",
        palette="mako",
        errorbar=None,
    )
    patches = ax.patches
    for patch, (_, row) in zip(patches, summary_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        yerr_lower = max(0.0, row["accuracy"] - row["ci_lower"])
        yerr_upper = max(0.0, row["ci_upper"] - row["accuracy"])
        ax.errorbar(x, row["accuracy"], yerr=[[yerr_lower], [yerr_upper]], fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by Condition and Prompt Style")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    ax.legend(title="Prompt Style")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=25, help="Number of base position pairs (total samples=2*pairs)")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    samples = generate_dataset(n_pairs=args.pairs, board_size=args.board_size, seed=args.seed)
    samples_df = pd.DataFrame(samples)
    dataset_path = Path("data/conditional_forgetting.csv")
    samples_df.to_csv(dataset_path, index=False)

    tokenizer, model = load_model(args.model)
    prompt_styles = ["plain", "reinforced"]
    responses_df = run_inference(samples_df, tokenizer, model, prompt_styles, max_new_tokens=args.max_new_tokens)
    responses_path = Path("results/responses.csv")
    responses_df.to_csv(responses_path, index=False)

    summary = summarize_by_group(responses_df, ["condition", "prompt_style"])
    summary_path = Path("results/metrics_table.csv")
    summary.to_csv(summary_path, index=False)

    plot_path = Path("results/plots/accuracy_by_condition.png")
    build_plot(summary, plot_path)

    metrics = {
        "summary": summary.to_dict(orient="records"),
        "default_bias": compute_default_bias(responses_df),
        "config": {
            "pairs": args.pairs,
            "board_size": args.board_size,
            "model": args.model,
            "seed": args.seed,
        },
    }

    # pairwise statistics for H2 and H3
    for style in prompt_styles:
        canon_rows = responses_df[(responses_df["condition"] == "canonical") & (responses_df["prompt_style"] == style)]
        counter_rows = responses_df[(responses_df["condition"] == "counterfactual") & (responses_df["prompt_style"] == style)]
        stats = proportions_ztest(
            success_a=int(canon_rows["is_correct"].sum()),
            size_a=int(len(canon_rows)),
            success_b=int(counter_rows["is_correct"].sum()),
            size_b=int(len(counter_rows)),
        )
        metrics[f"canon_vs_counter_{style}"] = stats

    counter_plain_rows = responses_df[(responses_df["condition"] == "counterfactual") & (responses_df["prompt_style"] == "plain")]
    counter_reinf_rows = responses_df[(responses_df["condition"] == "counterfactual") & (responses_df["prompt_style"] == "reinforced")]
    metrics["prompt_effect_counter"] = proportions_ztest(
        success_a=int(counter_reinf_rows["is_correct"].sum()),
        size_a=int(len(counter_reinf_rows)),
        success_b=int(counter_plain_rows["is_correct"].sum()),
        size_b=int(len(counter_plain_rows)),
    )

    metrics_path = Path("results/metrics.json")
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    config_path = Path("results/config.json")
    with config_path.open("w") as f:
        json.dump(
            {
                "python": os.sys.version,
                "packages": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "pandas": pd.__version__,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved dataset to {dataset_path}")
    print(f"Saved responses to {responses_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Metrics stored in {metrics_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
