"""Apply the official LMCache worker-initialization recipe only to this venv."""

import argparse
import difflib
import hashlib
import json
import sysconfig
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RELATIVE = Path("vllm/v1/worker/gpu_worker.py")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="save reviewable patch")
    parser.add_argument(
        "--restore", action="store_true", help="restore this venv's file"
    )
    args = parser.parse_args()
    original = (PROJECT / "upstream/vllm" / RELATIVE).read_bytes()
    modified = patched(original.decode()).encode()
    compile(modified, str(RELATIVE), "exec")
    if args.record:
        destination = PROJECT / "patches/vllm-worker-registration.patch"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(
                difflib.unified_diff(
                    original.decode().splitlines(keepends=True),
                    modified.decode().splitlines(keepends=True),
                    fromfile="a/" + str(RELATIVE),
                    tofile="b/" + str(RELATIVE),
                )
            )
        )
        print(
            json.dumps(
                {
                    "original_sha256": digest(original),
                    "patched_sha256": digest(modified),
                }
            )
        )
        return
    target = Path(sysconfig.get_paths()["purelib"]) / RELATIVE
    if not target.resolve().is_relative_to(PROJECT / ".venv"):
        raise ValueError(f"refusing to patch outside dedicated venv: {target}")
    current = target.read_bytes()
    if current not in (original, modified):
        raise ValueError("installed worker differs from both known states")
    desired = original if args.restore else modified
    target.write_bytes(desired)
    manifest = {
        "target": str(target),
        "original_sha256": digest(original),
        "patched_sha256": digest(modified),
        "current_sha256": digest(desired),
        "recipe": "upstream/lmcache/examples/blend_kv_v1/README.md",
    }
    (PROJECT / "state").mkdir(exist_ok=True)
    (PROJECT / "state/runtime_patch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
