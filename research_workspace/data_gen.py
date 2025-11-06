import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Coord = Tuple[int, int]  # (file_idx, rank_idx) zero-based


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def idx_to_square(idx: Coord) -> str:
    file_letter = chr(ord("a") + idx[0])
    rank_number = idx[1] + 1
    return f"{file_letter}{rank_number}"


def random_square(board_size: int) -> Coord:
    return random.randrange(board_size), random.randrange(board_size)


def knight_can_reach(src: Coord, dst: Coord) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return (dx, dy) in {(1, 2), (2, 1)}


def bishop_can_reach(src: Coord, dst: Coord) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return dx == dy and dx != 0


def rook_can_reach(src: Coord, dst: Coord) -> bool:
    return (src[0] == dst[0] or src[1] == dst[1]) and src != dst


def queen_can_reach(src: Coord, dst: Coord) -> bool:
    return bishop_can_reach(src, dst) or rook_can_reach(src, dst)


def king_can_reach(src: Coord, dst: Coord) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return max(dx, dy) == 1 and (dx != 0 or dy != 0)


def limited_bishop(src: Coord, dst: Coord, max_len: int = 3) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return dx == dy and 0 < dx <= max_len


def diagonal_knight(src: Coord, dst: Coord) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return (dx, dy) in {(0, 3), (3, 0), (0, 2), (2, 0)}  # teleport style


def manhattan_leaper(src: Coord, dst: Coord, step: int = 2) -> bool:
    dx = abs(src[0] - dst[0])
    dy = abs(src[1] - dst[1])
    return dx + dy == step and (dx != 0 or dy != 0)


PIECE_RULES = {
    "rook": rook_can_reach,
    "bishop": bishop_can_reach,
    "knight": knight_can_reach,
}


@dataclass
class CounterRule:
    name: str
    description: str
    evaluator: Callable[[Coord, Coord], bool]


COUNTER_RULES = {
    "rook": [
        CounterRule(
            name="rook_as_knight",
            description="The rook behaves like a knight, leaping in the usual L-shape.",
            evaluator=knight_can_reach,
        ),
        CounterRule(
            name="rook_diagonal_only",
            description="The rook may only move exactly two squares diagonally in any direction.",
            evaluator=lambda s, d: limited_bishop(s, d, max_len=2),
        ),
    ],
    "bishop": [
        CounterRule(
            name="bishop_as_rook",
            description="The bishop is constrained to straight files and ranks like a rook.",
            evaluator=rook_can_reach,
        ),
        CounterRule(
            name="bishop_short_king",
            description="The bishop now moves like a king, one square in any direction.",
            evaluator=king_can_reach,
        ),
    ],
    "knight": [
        CounterRule(
            name="knight_as_bishop",
            description="The knight glides like a bishop across any number of diagonal squares.",
            evaluator=bishop_can_reach,
        ),
        CounterRule(
            name="knight_taxicab",
            description="The knight travels exactly two squares by combining orthogonal steps (like a taxi making two short moves).",
            evaluator=lambda s, d: manhattan_leaper(s, d, step=2),
        ),
        CounterRule(
            name="knight_as_rook",
            description="The knight becomes a rook that can only move horizontally or vertically.",
            evaluator=rook_can_reach,
        ),
    ],
}


def generate_dataset(
    n_pairs: int,
    board_size: int = 6,
    seed: int = 42,
) -> List[dict]:
    """Generate paired canonical vs counterfactual samples with ground truth."""
    set_seed(seed)
    samples: List[dict] = []
    for base_id in range(n_pairs):
        piece = random.choice(list(PIECE_RULES))
        src = random_square(board_size)
        dst = random_square(board_size)
        while dst == src:
            dst = random_square(board_size)

        canonical_truth = PIECE_RULES[piece](src, dst)
        samples.append(
            {
                "sample_id": f"{base_id}_canon",
                "pair_id": base_id,
                "condition": "canonical",
                "piece": piece,
                "source": idx_to_square(src),
                "target": idx_to_square(dst),
                "rule_variant": "standard",
                "rule_text": f"Use the standard {piece} movement rules from chess.",
                "board_size": board_size,
                "correct_answer": "YES" if canonical_truth else "NO",
                "canonical_answer": "YES" if canonical_truth else "NO",
            }
        )

        counter_option = random.choice(COUNTER_RULES[piece])
        counter_truth = counter_option.evaluator(src, dst)
        samples.append(
            {
                "sample_id": f"{base_id}_counter",
                "pair_id": base_id,
                "condition": "counterfactual",
                "piece": piece,
                "source": idx_to_square(src),
                "target": idx_to_square(dst),
                "rule_variant": counter_option.name,
                "rule_text": counter_option.description,
                "board_size": board_size,
                "correct_answer": "YES" if counter_truth else "NO",
                "canonical_answer": "YES" if canonical_truth else "NO",
            }
        )
    return samples


def split_samples(
    samples: Sequence[dict],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Optional helper to split samples if needed."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]
