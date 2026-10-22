"""Record versions and require the wheel's algorithm to match pinned source."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig
from pathlib import Path

from apply_runtime_patch import (
    LM_GPU_RELATIVE,
    LM_RELATIVE,
    RELATIVE,
    patched,
    patched_gpu_connector,
    patched_lmcache,
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("use dedicated venv")
    lock = json.loads((PROJECT / "sources.lock.json").read_text())
    site = Path(sysconfig.get_paths()["purelib"])
    files = {}
    for relative, expected in lock["upstream"]["lmcache"]["files_sha256"].items():
        if not relative.startswith("lmcache/"):
            continue
        path = site / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        transforms = {
            str(LM_RELATIVE): patched_lmcache,
            str(LM_GPU_RELATIVE): patched_gpu_connector,
        }
        if relative in transforms:
            source = PROJECT / "upstream/lmcache" / relative
            expected = hashlib.sha256(
                transforms[relative](source.read_text()).encode()
            ).hexdigest()
        if actual != expected:
            raise ValueError(f"installed baseline algorithm differs: {relative}")
        files[relative] = actual
    packages = {}
    for name in ("vllm", "lmcache", "torch", "transformers", "flashinfer-python"):
        packages[name] = importlib.metadata.version(name)
    expected_versions = {
        "vllm": "0.11.0",
        "lmcache": "0.3.13",
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "flashinfer-python": "0.3.1",
    }
    if any(
        packages[n].split("+")[0] != version for n, version in expected_versions.items()
    ):
        raise ValueError(f"runtime versions changed: {packages}")
    expected_worker = patched((PROJECT / "upstream/vllm" / RELATIVE).read_text())
    if (site / RELATIVE).read_text() != expected_worker:
        raise ValueError("official worker registration recipe is not applied")
    for name in ("vllm", "lmcache", "torch"):
        origin = importlib.util.find_spec(name).origin
        if not Path(origin).resolve().is_relative_to(PROJECT / ".venv"):
            raise ValueError(f"import escaped venv: {origin}")
    data = {
        "python": platform.python_version(),
        "prefix": sys.prefix,
        "packages": packages,
        "verified_algorithm_files": files,
    }
    (PROJECT / "state").mkdir(exist_ok=True)
    (PROJECT / "state/runtime.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
