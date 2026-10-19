# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.promotion_calibration import CalibrationActions


def make_scheduler(**overrides):
    manager = MagicMock(spec=OffloadingManager)
    manager.lookup.return_value = 1
    extra = dict(
        promotion_policy="calibration_mixed",
        track_promotion_ttft=True,
        calibration_action_assignments={
            "chatcmpl-a": "restore",
            "chatcmpl-b": "recompute",
        },
    )
    extra.update(overrides)
    return OffloadingConnectorScheduler(
        SimpleNamespace(
            gpu_block_size=(4,),
            block_size_factor=3,
            get_manager=lambda: manager,
            extra_config=extra,
            kv_bytes_per_offloaded_block=1024,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True)
            ),
        )
    )


def test_frozen_mixed_actions_and_retry():
    s = make_scheduler()
    for name, expected in (("a", (12, True)), ("b", (0, False)), ("a", (12, True))):
        r = SimpleNamespace(
            request_id=f"chatcmpl-{name}-1234abcd",
            num_tokens=12,
            num_prompt_tokens=12,
            block_hashes=[BlockHash(str(i).encode()) for i in range(3)],
        )
        assert s.get_num_new_matched_tokens(r, 0) == expected
    assert [e["action"] for e in s.promotion_ttft_trace.events] == [
        "restore",
        "recompute",
        "restore",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(track_promotion_ttft=False),
        dict(calibration_action_assignments={}),
        dict(store_threshold=2),
    ],
)
def test_invalid_calibration_fails_closed(overrides):
    with pytest.raises(ValueError):
        make_scheduler(**overrides)


def test_unassigned_request_fails_closed():
    a = CalibrationActions({"chatcmpl-a": "restore"})
    for rid in ("chatcmpl-a", "chatcmpl-b-1234abcd", "chatcmpl-a-nothex00"):
        with pytest.raises(ValueError, match="Unassigned"):
            a.action(rid)


@pytest.mark.parametrize("field", ["admission_enabled", "guard_enabled"])
def test_mixed_forbids_adaptive_admission_or_guard(field):
    cfg = dict(
        admission_enabled=False,
        guard_enabled=False,
        store_s_per_block=0.001,
        restore_s_per_block=0.002,
        recompute_s_per_token=0.001,
    )
    cfg[field] = True
    with pytest.raises(ValueError, match="fixed admission"):
        make_scheduler(lifecycle_policy=cfg)
