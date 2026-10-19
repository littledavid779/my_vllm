# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host execution boundaries, never a CUDA/device duration measurement."""

import os
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get("VLLM_EXPERIMENTAL_RESTORE_HOST_TRACE") == "1"
_state = threading.local()


def mark(stage):
    if not ENABLED:
        return
    record = getattr(_state, "record", None)
    if record is None:
        return
    if len(record["marks"]) >= 96:
        record["overflow"] = True
        return
    record["marks"].append(
        dict(stage=stage, wall_s=time.perf_counter(), cpu_s=time.thread_time())
    )


@contextmanager
def execution_scope(scheduler_output):
    if not ENABLED:
        yield None
        return
    if getattr(_state, "record", None) is not None:
        raise RuntimeError("Nested host execution trace")
    record = dict(
        marks=[],
        overflow=False,
        scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        scheduled_requests=len(scheduler_output.num_scheduled_tokens),
        thread_id=threading.get_ident(),
    )
    _state.record = record
    mark("executor_dispatch")
    try:
        yield record
    finally:
        mark("end")
        _state.record = None
