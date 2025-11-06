from textwrap import dedent


BOARD_BRIEF = (
    "Coordinates follow chess notation with files a-f (left to right) and ranks 1-6 (bottom to top). "
    "Only one attacker piece and one target exist, so ignore obstructions."
)


def build_prompt(sample: dict, template: str) -> str:
    """Return a formatted prompt conditioned on the template name."""
    scenario = dedent(
        f"""
        You are analyzing a hypothetical chess variant on a {sample['board_size']}x{sample['board_size']} board.
        Attacker piece: {sample['piece'].title()} on {sample['source']}.
        Target piece: opponent piece on {sample['target']}.
        Variant rule: {sample['rule_text']}
        {BOARD_BRIEF}
        Decide whether the attacker can capture the target using ONLY the variant rule above.
        """
    ).strip()

    if template == "plain":
        instructions = (
            "State your reasoning in 1-2 sentences and finish with 'FINAL ANSWER: YES' or 'FINAL ANSWER: NO'."
        )
        return f"{scenario}\n\n{instructions}"

    if template == "reinforced":
        checklist = dedent(
            """
            Follow this checklist before answering:
            1. Explicitly restate the variant rule in your own words.
            2. Confirm you are ignoring all normal chess knowledge.
            3. Simulate the movement under the variant rule only.
            4. Output 'FINAL ANSWER: YES' or 'FINAL ANSWER: NO' on its own line.
            """
        ).strip()
        reminder = "If you cannot deduce the result with the variant rule alone, answer NO by default."
        return f"{scenario}\n\n{checklist}\n{reminder}"

    raise ValueError(f"Unknown template: {template}")
