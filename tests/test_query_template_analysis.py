"""Check the factorial interaction and prevent unpaired confidence intervals."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_query_template import interaction  # noqa: E402


def groups(values):
    return {
        m: [
            dict(
                repeat=rep,
                session_id="s",
                version=1,
                source_id="one",
                task_correct=value,
            )
        ]
        for m, value in values.items()
        for rep in (0,)
    }


def test_template_main_effect_without_interaction():
    data = groups(dict(none=False, question=False, template=True, both=True))
    result = interaction(data, draws=20)
    assert result == {"effect": 0, "source_cluster_95ci": [0, 0]}


def test_positive_interaction_when_only_joint_refresh_succeeds():
    data = groups(dict(none=False, question=False, template=False, both=True))
    result = interaction(data, draws=20)
    assert result == {"effect": 1, "source_cluster_95ci": [1, 1]}


def test_interaction_rejects_unpaired_or_duplicate_rows():
    data = groups(dict(none=False, question=False, template=False, both=True))
    data["none"] = []
    with pytest.raises(ValueError, match="unpaired"):
        interaction(data, draws=20)
    data["none"] = data["both"] * 2
    with pytest.raises(ValueError, match="duplicate"):
        interaction(data, draws=20)
