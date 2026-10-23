"""Check real GPU selection against a CPU oracle with unique synthetic scores."""

import argparse
import ast
import json
import os

from apply_query_control_patch import HELPER, MODES
from run_smoke import gpu_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    state = gpu_state(args.gpu)
    if state["pids"] or state["memory_mib"] > 100 or state["utilization"] > 5:
        raise RuntimeError("GPU busy before selector validation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    scope = {"torch": torch}
    exec(compile(ast.parse(HELPER), "query-control-gpu", "exec"), scope)
    select = scope["select_query_control"]
    cases = []
    for length in (37, 1024, 8044):
        scores = torch.arange(length, 0, -1).float()
        device_scores = scores.cuda()
        budget = max(1, int(length * 0.1))
        for mode in MODES:
            expected, expected_forced = select(
                scores, budget, mode, 19 if length >= 1024 else 0
            )
            actual, forced = select(
                device_scores, budget, mode, 19 if length >= 1024 else 0
            )
            if expected_forced != forced or set(expected.tolist()) != set(
                actual.tolist()
            ):
                raise ValueError("CPU/GPU selected positions differ")
            if not torch.equal(device_scores.cpu(), scores):
                raise ValueError("GPU selector changed scores")
            cases.append(
                {
                    "mode": mode,
                    "length": length,
                    "selected": actual.numel(),
                    "forced": forced,
                }
            )
    print(json.dumps({"passed": True, "gpu": args.gpu, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
