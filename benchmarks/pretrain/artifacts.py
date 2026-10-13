"""Standalone benchmark artifacts, exact boxed parsing and task-level bootstrap.

This module uses no training code. JSON records retain raw answers alongside
parsing decisions; compressed traces retain enough deltas to reconstruct every
native history by appending MASK positions and applying the stored changes.
"""

import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np


def sha256(path):
    """Hash a file without loading model weights into host memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a JSON artifact atomically; incomplete temporary files are ignored."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_npz(path, **arrays):
    """Write compressed, pickle-free arrays and atomically publish the archive."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def numeric_answer(text):
    """Parse integers, decimals, scientific notation or a simple LaTeX fraction.

Return an exact Fraction or None; arbitrary expressions are not evaluated.
The convention matches the existing boxed-number protocol, without importing it.
"""
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text.strip())
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    text = re.sub(r"\\(?:left|right)", "", text)
    fraction = re.fullmatch(r"(-?)\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        if not all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", x) for x in (numerator, denominator)):
            return None
        divisor = Fraction(denominator)
        return (-1 if sign else 1) * Fraction(numerator) / divisor if divisor else None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return Fraction(text)
    return None


def parse_answer(completion, gold):
    """Extract the final balanced boxed expression and expose its parsing status."""
    target = numeric_answer(gold.rsplit("####", 1)[-1])
    assert target is not None, "Unparseable GSM8K gold answer"
    starts = list(re.finditer(r"\\boxed\s*\{", completion))
    result = {"gold_number": str(target), "boxed_text": None, "prediction_number": None}
    if not starts:
        return result | {"parsing_status": "missing_boxed", "reward": 0}
    start = starts[-1].end()
    depth, end = 1, start
    while end < len(completion) and depth:
        depth += (completion[end] == "{") - (completion[end] == "}")
        end += 1
    if depth:
        return result | {"parsing_status": "unclosed_boxed", "reward": 0}
    result["boxed_text"] = completion[start:end - 1]
    prediction = numeric_answer(result["boxed_text"])
    result["prediction_number"] = str(prediction) if prediction is not None else None
    return result | {
        "parsing_status": "parsed" if prediction is not None else "invalid_number",
        "reward": int(prediction is not None and prediction == target),
    }


def pass_at_k(correct, total, k):
    """Unbiased combinatorial estimator for a task with total independent draws."""
    return 1.0 if total - correct < k else 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def trace_arrays(histories, observed, mask_id, block_size):
    """Encode native histories and reveal confidence without storing vocab logits.

history_lengths defines the extent after each step. Initial tokens are stored
verbatim; extension positions start as MASK. Deltas use absolute padded sequence
positions and zero-based sample rows. Observer entries correspond one-to-one to
histories after the initial state; confidence uses native logits softmax dtype.
"""
    lengths = np.asarray([x.shape[1] for x in histories], dtype=np.int32)
    states = [x.cpu().numpy().astype(np.int32) for x in histories]
    assert len(observed) == len(states) - 1
    changes, reveals = [], []
    offsets, blocks, local_steps = [], [], []
    block, local_step = -1, 0
    initial_length = int(lengths[0])
    for step, (previous, current, entry) in enumerate(zip(states, states[1:], observed), 1):
        if current.shape[1] != previous.shape[1]:
            block += 1
            local_step = 0
        offset = initial_length + block * block_size
        offsets.append(offset)
        blocks.append(block)
        local_steps.append(local_step)
        local_step += 1
        before = np.full(current.shape, mask_id, dtype=np.int32)
        before[:, :previous.shape[1]] = previous
        rows, positions = np.nonzero(current != before)
        changes.append(np.column_stack((np.full(len(rows), step), rows, positions, current[rows, positions])))
        revealed, probability, _ = entry
        rows, positions = np.nonzero(revealed.cpu().numpy())
        probs = probability.cpu().numpy()[rows, positions]
        reveals.append(np.column_stack((np.full(len(rows), step), rows, positions + offset, probs)))
    delta = np.concatenate(changes).astype(np.int32) if changes else np.empty((0, 4), dtype=np.int32)
    revealed = np.concatenate(reveals) if reveals else np.empty((0, 4))
    confidence = revealed[:, 3].astype(np.float32)
    return {
        "initial_ids": states[0], "history_lengths": lengths, "delta": delta,
        "reveal_index": revealed[:, :3].astype(np.int32), "confidence": confidence,
        "log_confidence": np.log(np.maximum(confidence, np.finfo(np.float32).tiny)),
        "transfer_quota": np.stack([entry[2].cpu().numpy() for entry in observed]).astype(np.int32),
        "block_index": np.asarray(blocks, dtype=np.int32),
        "step_in_block": np.asarray(local_steps, dtype=np.int32),
        "block_offset": np.asarray(offsets, dtype=np.int32),
        "mask_id": np.asarray(mask_id, dtype=np.int32),
    }


def reconstruct_trace(arrays):
    """Reconstruct native final IDs for an artifact consistency check."""
    state = arrays["initial_ids"].copy()
    for step, length in enumerate(arrays["history_lengths"][1:], 1):
        extended = np.full((len(state), int(length)), int(arrays["mask_id"]), dtype=np.int32)
        extended[:, :state.shape[1]] = state
        change = arrays["delta"][arrays["delta"][:, 0] == step]
        extended[change[:, 1], change[:, 2]] = change[:, 3]
        state = extended
    return state


def aggregate(records, samples, ks):
    """Aggregate only complete tasks for Pass@k; retain counts for all saved draws."""
    groups = {}
    for record in records:
        groups.setdefault(record["example_id"], []).append(record)
    complete = {key: value for key, value in groups.items() if len(value) == samples}
    scores = np.asarray([
        [pass_at_k(sum(r["reward"] for r in complete[key]), samples, k) for k in ks]
        for key in sorted(complete)
    ], dtype=np.float64).reshape(-1, len(ks))
    means = scores.mean(axis=0).tolist() if len(scores) else [0.0] * len(ks)
    stats = {
        "completed_problems": len(complete), "saved_samples": len(records),
        "pass_at_k": dict(zip(map(str, ks), means)),
        "parse_failure_rate": sum(r["parsing_status"] != "parsed" for r in records) / max(1, len(records)),
        "mean_answer_tokens": sum(r["answer_length"] for r in records) / max(1, len(records)),
    }
    for name in ("has_eos", "has_mask", "has_pad"):
        stats[name + "_rate"] = sum(r[name] for r in records) / max(1, len(records))
    return stats, scores, np.asarray(sorted(complete), dtype=np.int32)


def bootstrap(scores, repetitions, seed, confidence):
    """Task-level percentile bootstrap with shared resampling indices across k.

Intervals condition on the collected generation draws. They are individual
intervals, not simultaneous bounds and not estimates from repeated model seeds.
"""
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, scores.shape[1]), dtype=np.float64)
    for start in range(0, repetitions, 256):
        count = min(256, repetitions - start)
        indices = rng.integers(len(scores), size=(count, len(scores)))
        draws[start:start + count] = scores[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    bounds = np.quantile(draws, [alpha, 1 - alpha], axis=0)
    return draws, bounds
