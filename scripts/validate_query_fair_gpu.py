"""Verify the actual GPU selectors against independent fixed-boundary cases."""

import argparse
import json
import os

from apply_query_fair_patch import ALL_HELPERS, MODES
from run_smoke import gpu_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    state = gpu_state(args.gpu)
    if state["pids"] or state["memory_mib"] > 100 or state["utilization"] > 5:
        raise RuntimeError("GPU busy before validation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    scope = {"torch": torch}
    exec(compile(ALL_HELPERS, "query-fair-gpu", "exec"), scope)
    select = scope["select_query_fair"]
    cases = []
    for length in (37, 1024, 8044):
        scores = torch.arange(length, 0, -1).float()
        device_scores = scores.cuda()
        budget = max(1, int(length * 0.1))
        qlen = 19 if length >= 1024 else 0
        starts = [200, 240] if length >= 1024 else []
        for mode in MODES:
            actual, forced = select(device_scores, budget, mode, qlen, starts)
            got = set(actual.tolist())
            if length < 1024:
                expected = set(range(budget))
            elif mode == "boundary":
                # First nonfirst chunk has only 40 tokens: it is exhausted.
                expected = set(range(200, 240)) | set(
                    range(240, 240 + budget - qlen - 40)
                )
                expected |= set(range(length - qlen, length))
            else:
                reserve = qlen if mode == "query" else 64
                expected = set(range(budget - reserve)) | set(
                    range(length - reserve, length)
                )
            if got != expected or actual.numel() != budget:
                raise ValueError(f"incorrect GPU selector: {mode}/{length}")
            if not torch.equal(scores, device_scores.cpu()):
                raise ValueError("selector mutated input scores")
            cases.append(
                {"mode": mode, "length": length, "selected": len(got), "forced": forced}
            )
    print(json.dumps({"passed": True, "gpu": args.gpu, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
