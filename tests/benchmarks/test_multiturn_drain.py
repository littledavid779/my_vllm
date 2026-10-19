# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client queue exhaustion must not discard active conversation turns."""

import asyncio
import importlib
import queue
from pathlib import Path
from threading import Event

import pytest


@pytest.mark.parametrize(
    "lengths,cap,expected",
    [([4, 6], None, 5), ([1, 3], None, 3), ([4, 6], 2, 2), ([], None, 0)],
)
def test_client_drains_active_conversations(monkeypatch, lengths, cap, expected):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/multi_turn")
    )
    bench = importlib.import_module("benchmark_serving_multi_turn")
    tasks, results, completed = queue.Queue(), queue.Queue(), queue.Queue()
    for i, length in enumerate(lengths):
        tasks.put(
            (
                str(i),
                [
                    {"role": "user" if n % 2 == 0 else "assistant", "content": "x"}
                    for n in range(length)
                ],
            )
        )
    tasks.put((None, None))
    seen = []

    async def fake_send(session, client, conv, messages, turn, *unused):
        assert messages[turn - 1]["role"] == "user"
        seen.append((conv, turn))
        if len(messages) == turn:
            messages.append({"role": "assistant", "content": "answer"})
        return (conv, turn)

    monkeypatch.setattr(bench, "send_turn", fake_send)
    args = bench.ClientArgs(
        seed=0,
        max_num_requests=cap,
        skip_first_turn=False,
        max_turns=None,
        max_active_conversations=8,
        verbose=False,
        print_content=False,
        verify_output=False,
        preserve_dataset_assistant_history=True,
        conversation_sampling=bench.ConversationSampling.ROUND_ROBIN,
        request_rate=0,
        max_retries=0,
    )
    asyncio.run(
        bench.client_main(args, None, 0, None, Event(), tasks, results, completed)
    )
    assert len(seen) == results.qsize() == expected
    assert len(set(seen)) == expected
    if cap is None:
        assert set(seen) == {
            (str(i), turn)
            for i, length in enumerate(lengths)
            for turn in range(1, length + 1, 2)
        }
        assert completed.qsize() == len(lengths) + 1
