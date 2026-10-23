"""Check strict GPU region exclusion against an independent ranking oracle."""

import argparse
import json
import os

from apply_query_template_patch import HELPER, MODES
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
    exec(compile(HELPER, "query-template-gpu", "exec"), scope)
    cases = []
    for length in (37, 1024, 8044):
        for scores_kind in ("tail_highest", "permuted"):
            cpu = torch.arange(length).float()
            if scores_kind == "permuted":
                cpu = torch.randperm(
                    length, generator=torch.Generator().manual_seed(231026)
                ).float()
            device_scores = cpu.cuda()
            budget = max(1, int(length * 0.1))
            qlen, tlen = (9, 9) if length >= 1024 else (0, 0)
            q = set(range(length - qlen - tlen, length - tlen))
            t = set(range(length - tlen, length))
            for mode in MODES:
                mandatory = set()
                allowed = set(range(length))
                if length >= 1024 and mode != "topk":
                    allowed -= q | t
                    if mode in ("question", "both"):
                        mandatory |= q
                    if mode in ("template", "both"):
                        mandatory |= t
                ranking = sorted(allowed, key=lambda i: float(cpu[i]), reverse=True)
                expected = mandatory | set(ranking[: budget - len(mandatory)])
                actual, forced = scope["select_query_template"](
                    device_scores, budget, mode, qlen, tlen
                )
                if (
                    set(actual.tolist()) != expected
                    or actual.numel() != budget
                    or forced != len(mandatory)
                ):
                    raise ValueError(
                        f"GPU selector differs from oracle: {mode}/{length}"
                    )
                if not torch.equal(device_scores.cpu(), cpu):
                    raise ValueError("selector mutated scores")
                cases.append(
                    {
                        "mode": mode,
                        "length": length,
                        "scores": scores_kind,
                        "selected": budget,
                        "forced": forced,
                    }
                )
    print(json.dumps({"passed": True, "gpu": args.gpu, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
