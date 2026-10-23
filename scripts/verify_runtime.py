"""Record versions and require the wheel's algorithm to match pinned source."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig
from pathlib import Path

from apply_precheck_kv_patch import RELATIVE as PRECHECK_RELATIVE
from apply_precheck_kv_patch import commit_precheck
from apply_query_control_patch import query_controls
from apply_query_fair_patch import fair_controls
from apply_query_tail_patch import reserve_query_tail
from apply_runtime_patch import (
    LM_GPU_RELATIVE,
    LM_RELATIVE,
    RELATIVE,
    patched,
    patched_gpu_connector,
    patched_lmcache,
)
from apply_sparse_index_patch import (
    BACKEND_RELATIVE,
    SPARSE_RELATIVE,
    copied_indices,
    workspace_argument,
)
from apply_strict_causal_patch import strict_backend, strict_metadata

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-sparse-correctness-fix",
        action="store_true",
        dest="allow_sparse_index_fix",
    )
    parser.add_argument("--allow-strict-causal", action="store_true")
    parser.add_argument("--allow-precheck-kv-write", action="store_true")
    parser.add_argument("--allow-query-tail", action="store_true")
    parser.add_argument("--allow-query-controls", action="store_true")
    parser.add_argument("--allow-query-fair", action="store_true")
    args = parser.parse_args()
    if args.allow_query_fair:
        if args.allow_query_controls:
            parser.error("choose one query patch")
        args.allow_query_controls = True
    if args.allow_query_controls and (
        args.allow_query_tail
        or not (args.allow_strict_causal and args.allow_precheck_kv_write)
    ):
        parser.error(
            "query controls require strict/common correction and no query-tail patch"
        )
    if args.allow_query_tail and not args.allow_precheck_kv_write:
        parser.error("query-tail requires the common pre-check KV correction")
    if args.allow_strict_causal and args.allow_sparse_index_fix:
        parser.error("strict and block-mask diagnostics cannot be mixed")
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
        if args.allow_sparse_index_fix:
            transforms[str(SPARSE_RELATIVE)] = copied_indices
            transforms[str(BACKEND_RELATIVE)] = workspace_argument
        if args.allow_strict_causal:
            transforms[str(SPARSE_RELATIVE)] = strict_metadata
            transforms[str(BACKEND_RELATIVE)] = strict_backend
        if args.allow_precheck_kv_write:
            transforms[str(PRECHECK_RELATIVE)] = commit_precheck
        if args.allow_query_tail:
            transforms[str(PRECHECK_RELATIVE)] = lambda source: reserve_query_tail(
                commit_precheck(source)
            )
        if args.allow_query_controls:
            transforms[str(PRECHECK_RELATIVE)] = lambda source: (
                fair_controls if args.allow_query_fair else query_controls
            )(commit_precheck(source))
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
        "sparse_correctness_fix": args.allow_sparse_index_fix,
        "strict_causal": args.allow_strict_causal,
        "precheck_kv_write": args.allow_precheck_kv_write,
        "query_tail_patch": args.allow_query_tail,
        "query_controls_patch": args.allow_query_controls,
        "query_fair_patch": args.allow_query_fair,
    }
    (PROJECT / "state").mkdir(exist_ok=True)
    (PROJECT / "state/runtime.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
