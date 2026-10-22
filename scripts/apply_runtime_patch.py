"""Apply explicit integration compatibility patches only to this venv."""

import argparse
import ast
import difflib
import hashlib
import json
import os
import sysconfig
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RELATIVE = Path("vllm/v1/worker/gpu_worker.py")
LM_RELATIVE = Path("lmcache/integration/vllm/vllm_v1_adapter.py")
LM_GPU_RELATIVE = Path("lmcache/v1/gpu_connector.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched(text):
    early = "    ensure_kv_transfer_initialized(vllm_config)\n"
    load = "            self.model_runner.load_model(eep_scale_up=eep_scale_up)\n"
    if text.count(early) != 1 or text.count(load) != 1:
        raise ValueError("unexpected upstream worker source; refusing heuristic patch")
    text = text.replace(
        early, "    # LMCache recipe: initialize after model registration.\n"
    )
    return text.replace(
        load,
        load
        + "\n"
        + "        from lmcache.v1.compute.models.utils import VLLMModelTracker\n"
        "        from lmcache.integration.vllm.utils import ENGINE_NAME\n\n"
        "        VLLMModelTracker.register_model("
        "ENGINE_NAME, self.model_runner.model)\n"
        "        ensure_kv_transfer_initialized(self.vllm_config)\n",
    )


def patched_lmcache(text):
    old = "            token_ids = request.all_token_ids\n"
    if text.count(old) != 1:
        raise ValueError("unexpected LMCache request-token adapter source")
    return text.replace(old, "            token_ids = list(request.all_token_ids)\n")


def patched_gpu_connector(text):
    # C++ declares use_mla=false, but pybind exposes seven required arguments.
    # Two full-attention buffer-path call sites still pass only six arguments.
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "single_layer_kv_transfer"
        and len(node.args) == 6
    ]
    if len(calls) != 2:
        raise ValueError("unexpected six-argument transfer call sites")
    lines = text.splitlines(keepends=True)
    for node in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        previous = lines[node.end_lineno - 2]
        indent = previous[: len(previous) - len(previous.lstrip())]
        lines.insert(
            node.end_lineno - 1, indent + "False,  # use_mla (full attention)\n"
        )
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="save reviewable patch")
    parser.add_argument(
        "--restore", action="store_true", help="restore this venv's file"
    )
    args = parser.parse_args()
    plans = (
        ("vllm", RELATIVE, patched, "vllm-worker-registration.patch"),
        ("lmcache", LM_RELATIVE, patched_lmcache, "lmcache-constant-list.patch"),
        (
            "lmcache",
            LM_GPU_RELATIVE,
            patched_gpu_connector,
            "lmcache-transfer-args.patch",
        ),
    )
    pending, files = [], []
    for owner, relative, transform, patch_name in plans:
        original = (PROJECT / "upstream" / owner / relative).read_bytes()
        modified = transform(original.decode()).encode()
        compile(modified, str(relative), "exec")
        if args.record:
            destination = PROJECT / "patches" / patch_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                "".join(
                    difflib.unified_diff(
                        original.decode().splitlines(keepends=True),
                        modified.decode().splitlines(keepends=True),
                        fromfile="a/" + str(relative),
                        tofile="b/" + str(relative),
                    )
                )
            )
            continue
        target = Path(sysconfig.get_paths()["purelib"]) / relative
        if not target.resolve().is_relative_to(PROJECT / ".venv"):
            raise ValueError(f"refusing to patch outside dedicated venv: {target}")
        if target.read_bytes() not in (original, modified):
            raise ValueError(f"installed file differs from known states: {relative}")
        desired = original if args.restore else modified
        pending.append((target, desired))
        files.append(
            {
                "target": str(target),
                "original_sha256": digest(original),
                "patched_sha256": digest(modified),
                "current_sha256": digest(desired),
                "patch": patch_name,
            }
        )
    if args.record:
        print("Recorded reviewable integration patches")
        return
    # Validate every target first. Break uv's hardlinks before changing files.
    for target, desired in pending:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            stream.write(desired)
            temporary = Path(stream.name)
        temporary.chmod(target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    manifest = {"files": files}
    (PROJECT / "state").mkdir(exist_ok=True)
    (PROJECT / "state/runtime_patch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
