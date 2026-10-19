# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print an opt-in KV transfer configuration with explicit cost priors.

Costs on the command line are milliseconds. No performance priors are assumed.
Use a separate calibration split; do not tune these on evaluation outcomes.
"""

import argparse
import json
from dataclasses import asdict

from vllm.v1.kv_offload.lifecycle_policy import LifecyclePolicyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["all-block", "admission", "promotion", "full"], default="full"
    )
    parser.add_argument("--cpu-bytes", type=int, required=True)
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="Tokens per offloaded block; must match calibration.",
    )
    parser.add_argument("--store-ms-per-block", type=float, required=True)
    parser.add_argument("--restore-ms-per-block", type=float, required=True)
    parser.add_argument("--recompute-ms-per-token", type=float, required=True)
    parser.add_argument("--restore-fixed-ms", type=float, default=0)
    parser.add_argument("--capacity-ms-per-block", type=float, default=0)
    parser.add_argument("--disable-guard", action="store_true")
    parser.add_argument(
        "--cold-probe-every",
        type=int,
        default=0,
        help="0 disables finite NORMAL-state exploration",
    )
    parser.add_argument("--cold-probe-min-blocks", type=int, default=16)
    parser.add_argument("--cold-probe-max-blocks", type=int, default=64)
    parser.add_argument("--cold-probe-budget-blocks", type=int, default=128)
    parser.add_argument("--cold-probe-budget-ms", type=float, default=100)
    args = parser.parse_args()
    if args.cpu_bytes <= 0 or args.block_size <= 0:
        parser.error("cpu-bytes and block-size must be positive")
    try:
        policy = LifecyclePolicyConfig(
            store_s_per_block=args.store_ms_per_block / 1000,
            restore_s_per_block=args.restore_ms_per_block / 1000,
            recompute_s_per_token=args.recompute_ms_per_token / 1000,
            restore_fixed_s=args.restore_fixed_ms / 1000,
            capacity_s_per_block=args.capacity_ms_per_block / 1000,
            admission_enabled=args.arm != "promotion",
            guard_enabled=args.arm == "full" and not args.disable_guard,
            cold_probe_every=args.cold_probe_every,
            cold_probe_min_blocks=args.cold_probe_min_blocks,
            cold_probe_max_blocks=args.cold_probe_max_blocks,
            cold_probe_budget_blocks=args.cold_probe_budget_blocks,
            cold_probe_budget_s=args.cold_probe_budget_ms / 1000,
        )
    except ValueError as exc:
        parser.error(str(exc))
    extra = {
        "cpu_bytes_to_use": args.cpu_bytes,
        "block_size": args.block_size,
        "eviction_policy": "lru",
        "store_threshold": 0,
        "track_lifecycle_stats": True,
        "track_cost_events": True,
        "promotion_policy": "always_restore"
        if args.arm in ("all-block", "admission")
        else "cost_aware",
    }
    if args.arm != "all-block":
        extra["lifecycle_policy"] = asdict(policy)
    print(
        json.dumps(
            {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": extra,
            }
        )
    )


if __name__ == "__main__":
    main()
