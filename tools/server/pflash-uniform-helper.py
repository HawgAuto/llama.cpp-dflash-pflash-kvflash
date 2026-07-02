#!/usr/bin/env python3
"""Reference PFlash helper for the experimental keep-index protocol.

Usage:
    pflash-uniform-helper.py <request-json> <pflash-drafter-gpu>

The request JSON contains tokens, keep_ratio, preserve_prefix, preserve_suffix,
and optional structural_indices. The helper prints JSON with keep_indices and
optional scores. The server validates indices and maps them back to token ids.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys


def fail(message: str) -> int:
    print(f"pflash-uniform-helper: {message}", file=sys.stderr)
    return 2


def edge_indices(n_src: int, n_keep: int, prefix: int, suffix: int) -> set[int]:
    keep: set[int] = set()
    for i in range(min(prefix, n_src)):
        keep.add(i)
    start = max(0, n_src - suffix)
    for i in range(start, n_src):
        keep.add(i)
    return keep


def uniform_keep_indices(tokens: list[int], keep_ratio: float, prefix: int, suffix: int, structural: list[int]) -> list[int]:
    n_src = len(tokens)
    if n_src < 4:
        return list(range(n_src))

    n_keep = max(2, math.ceil(n_src * keep_ratio))
    if n_keep >= n_src:
        return list(range(n_src))

    keep = edge_indices(n_src, n_keep, prefix, suffix)
    for idx in structural:
        if 0 <= idx < n_src:
            keep.add(idx)

    remaining = max(0, n_keep - len(keep))
    middle_begin = min(prefix, n_src)
    middle_end = max(middle_begin, n_src - suffix)
    n_middle_src = middle_end - middle_begin
    if remaining > 0 and n_middle_src > 0:
        for i in range(remaining):
            idx = middle_begin + ((i + 1) * n_middle_src) // (remaining + 1)
            keep.add(min(n_src - 1, idx))

    # If structural preservation overshot the budget, keep it anyway. The server
    # decides whether to accept oversize structural output.
    return sorted(keep)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        return fail("expected: <request-json> <pflash-drafter-gpu>")

    req_path = pathlib.Path(argv[1])
    if not req_path.is_file():
        return fail(f"request file not found: {req_path}")

    try:
        req = json.loads(req_path.read_text())
        tokens = [int(x) for x in req["tokens"]]
        keep_ratio = float(req["keep_ratio"])
        prefix = int(req.get("preserve_prefix", 64))
        suffix = int(req.get("preserve_suffix", 64))
        structural = [int(x) for x in req.get("structural_indices", [])]
    except Exception as exc:
        return fail(str(exc))

    if not tokens:
        return fail("token list is empty")
    if not (0.0 < keep_ratio < 1.0):
        return fail("keep-ratio must be greater than 0 and less than 1")
    if prefix < 0 or suffix < 0:
        return fail("preserve_prefix/preserve_suffix must be non-negative")

    keep_indices = uniform_keep_indices(tokens, keep_ratio, prefix, suffix, structural)
    scores = [1.0 if i in set(keep_indices) else 0.0 for i in range(len(tokens))]
    print(json.dumps({"keep_indices": keep_indices, "scores": scores}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
