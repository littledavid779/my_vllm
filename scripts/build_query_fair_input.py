"""Previously unevaluated source split for query-fair selector screening."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from public_oracle import FACT, ROOMS, babi_oracle

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = Path("/home/ljc/vllm/datasets/authoritative/suffix_public")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def presets():
    tree = ast.parse((SOURCES / "babilong_prompts.py").read_text())
    return ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "DEFAULT_PROMPTS" for t in n.targets
            )
        )
    )


def candidates(text, question, task, target):
    result = []
    for match in FACT.finditer(text):
        fact = match[0]
        if task == "qa9":
            person, room = re.search(r"Is (\w+) in the (\w+)\?", question).groups()
            if not fact.startswith(person) or f"the {room}" not in fact:
                continue
            choices = [
                f"{person} is in the {room}.",
                f"{person} is no longer in the {room}.",
            ]
        elif any(f"the {r}" in fact for r in ROOMS):
            choices = [re.sub("|".join(ROOMS), r, fact) for r in ROOMS]
        else:
            continue
        valid = []
        for alternative in choices:
            updated = text[: match.start()] + alternative + text[match.end() :]
            value = babi_oracle(FACT.findall(updated), question, task)
            if value is not None and value != target:
                valid.append((alternative, value))
        if valid:
            result.append((match, valid))
    return result


def main():
    global SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=16)
    parser.add_argument("--source-start", type=int, default=66)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument(
        "--exclude-input", type=Path, default=PROJECT / "data/round2_screen.json"
    )
    parser.add_argument("--edits", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output")
    SOURCES = args.sources.resolve()
    prior = json.loads(args.exclude_input.read_text())
    prior_ids = {s["source_id"] for s in prior["sessions"]}
    if args.source_start < 66:
        raise ValueError("new split starts at source index 66 or later")
    from transformers import AutoTokenizer

    model = "/hole0/model_info/model_zoo/downloads/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    separator_text = "# # KV_SEGMENT_7a9c # #"
    separator = tokenizer.encode(separator_text)[1:]
    assert separator
    pp = presets()
    sessions, exclusions, audit = [], [], {}
    for task in ("qa9", "qa2", "qa3"):
        source = SOURCES / f"{task}_4k.json"
        samples = json.loads(source.read_text())
        for item in samples:
            assert (
                babi_oracle(FACT.findall(item["input"]), item["question"], task)
                == item["target"]
            )
        audit[task] = {"source_labels_agreed": len(samples), "sha256": digest(source)}
        chosen = 0
        # Never use sources from earlier rounds, regardless of model correctness.
        for index, item in enumerate(
            samples[args.source_start :], start=args.source_start
        ):
            if f"{task}_{index:03d}" in prior_ids:
                raise ValueError("new split overlaps earlier model-evaluation sources")
            if chosen == args.per_task:
                break
            original, question, target = item["input"], item["question"], item["target"]
            eligible = candidates(original, question, task, target)
            if not eligible:
                exclusions.append([task, index, "no answer-changing single fact edit"])
                continue
            # Earliest causally effective fact, chosen without seeing model responses.
            match, valid = eligible[0]
            fact_index = next(
                i
                for i, m in enumerate(FACT.finditer(original))
                if m.start() == match.start()
            )
            facts = FACT.findall(original)
            background = FACT.sub("", original)
            background_ids = encode(background)
            noise = tokenizer.decode(
                (background_ids * (8000 // len(background_ids) + 2))[:7600]
            )
            early = "\n".join(facts) + "\n\n" + noise
            assert FACT.findall(early) == facts
            for placement, text in (("natural4k", original), ("early8k", early)):
                matches = list(FACT.finditer(text))
                edit = matches[fact_index]
                pool = [(edit[0], target), *valid]
                sid = f"{task}_{index:03d}_{placement}"
                instruction = (
                    pp[task]["instruction"]
                    + "\n\n"
                    + pp[task]["examples"]
                    + "\n\nFor the final question, give only one answer word: "
                    + ("yes or no." if task == "qa9" else "the location.")
                    + " Do not explain.\n\nRecord key: "
                    + sid
                    + "\n<context>\n"
                )
                rendered = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": instruction
                            + text
                            + "\n</context>\n"
                            + f"End record key: {sid}\nQuestion: {question}",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                offset = rendered.index(text)
                a = encode(rendered[: offset + edit.start()])
                c = encode(rendered[offset + edit.end() :])
                replacements = [pool[v % len(pool)][0] for v in range(args.edits + 1)]
                bs = [
                    encode(r + f" [record {sid} revision {v:04d}]\n")
                    for v, r in enumerate(replacements)
                ]
                budget = max(map(len, bs))
                padding = encode("\n")
                assert len(padding) == 1
                versions = []
                for v, (r, b) in enumerate(zip(replacements, bs, strict=True)):
                    answer = babi_oracle(
                        FACT.findall(text[: edit.start()] + r + text[edit.end() :]),
                        question,
                        task,
                    )
                    assert answer == pool[v % len(pool)][1]
                    b = b + padding * (budget - len(b))
                    versions.append(
                        {
                            "token_ids": a + separator + b + separator + c,
                            "new_chunk_ids": b,
                            "expected_value": answer,
                            "definition": r,
                        }
                    )
                for segment in (a, c, *[v["new_chunk_ids"] for v in versions]):
                    if any(
                        segment[i : i + len(separator)] == separator
                        for i in range(len(segment) - len(separator) + 1)
                    ):
                        raise ValueError("source contains separator")
                sessions.append(
                    {
                        "session_id": sid,
                        "source_id": f"{task}_{index:03d}",
                        "family": f"babilong_{task}",
                        "placement": placement,
                        "prompt_tokens": len(versions[0]["token_ids"]),
                        "editable_token_span": [len(a), len(a) + budget],
                        "suffix_tokens": len(c),
                        "versions": versions,
                        "metadata": {
                            "source_index": index,
                            "question": question,
                            "original_facts": facts,
                            "edited_fact_index": fact_index,
                            "official_target": target,
                        },
                    }
                )
            chosen += 1
        if chosen != args.per_task:
            raise ValueError(f"only {chosen} eligible {task} source cases")
    warm = [
        encode(f"Disjoint warmup {name} archive.\n" * 100)
        for name in ("amber", "silver", "violet")
    ]
    data = {
        "model": model,
        "sessions": sessions,
        "warmup_seed_ids": warm[0] + separator + warm[1] + separator + warm[2],
        "warmup_edit_ids": warm[2] + separator + warm[0] + separator + warm[1],
        "pilot_provenance": {
            "separator_ids": separator,
            "separator_text": separator_text,
        },
        "source_audit": audit,
        "exclusions": exclusions,
        "protocol": {
            "independent_sources": 3 * args.per_task,
            "edits": args.edits,
            "selection": (
                "first eligible sources at/after 66; earliest effective edit; "
                "no model-quality filtering"
            ),
            "quality_filtering": False,
            "exclude_input_sha256": digest(args.exclude_input),
            "source_start": args.source_start,
            "previously_evaluated_source_ids": sorted(prior_ids),
            "prompt_source_sha256": digest(SOURCES / "babilong_prompts.py"),
            "early8k": "facts moved before 7600 background tokens; facts removed",
            "caveat": "Derived task; not official BABILong benchmark scores.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    lock = {
        "input_sha256": digest(args.output),
        "protocol": data["protocol"],
        "source_audit": audit,
        "builder_sha256": digest(Path(__file__)),
        "oracle_sha256": digest(PROJECT / "scripts/public_oracle.py"),
    }
    args.output.with_suffix(".lock.json").write_text(json.dumps(lock, indent=2))
    print(
        json.dumps(
            {
                "chains": len(sessions),
                "sources": 3 * args.per_task,
                "max_tokens": max(s["prompt_tokens"] for s in sessions),
            }
        )
    )


if __name__ == "__main__":
    main()
