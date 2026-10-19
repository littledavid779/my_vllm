# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, experiment-only per-request action assignments. Not a policy model."""

import re


class CalibrationActions:
    def __init__(self, assignments):
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError("calibration_mixed requires nonempty action assignments")
        if any(
            not isinstance(k, str)
            or not k.startswith("chatcmpl-")
            or v not in ("restore", "recompute")
            for k, v in assignments.items()
        ):
            raise ValueError("Invalid calibration action assignment")
        self.assignments = dict(assignments)

    def action(self, request_id: str) -> str:
        # The correlated benchmark supplies external IDs; engine appends 8 hex.
        match = re.fullmatch(r"(.+)-[0-9a-f]{8}", request_id)
        if match is None or match[1] not in self.assignments:
            raise ValueError(f"Unassigned calibration request: {request_id}")
        return self.assignments[match[1]]
