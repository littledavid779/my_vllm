"""Frozen exploratory source panel with original, paraphrased and label prompts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from build_query_fair_input import SOURCES, candidates, presets
from public_oracle import FACT, ROOMS, babi_oracle

PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "paraphrase", "labels")
COUNTS = {"qa9": 16, "qa2": 8, "qa3": 8}
ROOM_LABELS = dict(
    zip(
        sorted(ROOMS),
        ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"),
        strict=True,
    )
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paraphrase(question, task):
    question = question.strip()
    if task == "qa9":
        person, room = re.fullmatch(r"Is (\w+) in the (\w+)\?", question).groups()
        return f"Is it true that {person} is in the {room}?"
    if task == "qa2":
        obj = re.fullmatch(r"Where is the (\w+)\?", question)[1]
        return f"Which room is the {obj} in?"
    obj, room = re.fullmatch(
        r"Where was the (\w+) before the (\w+)\?", question
    ).groups()
    return f"Before the {room}, where was the {obj}?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new input path")
    from transformers import AutoTokenizer

    prior_path = PROJECT / "data/query_fair_v1.json"
    prior = json.loads(prior_path.read_text())
    model = prior["model"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    sep = prior["pilot_provenance"]["separator_ids"]
    pp, sessions, source_audit = presets(), [], {}
    for task, count in COUNTS.items():
        ids = sorted(
            {
                s["source_id"]
                for s in prior["sessions"]
                if s["family"] == "babilong_" + task
            }
        )[:count]
        assert len(ids) == count
        source = SOURCES / f"{task}_4k.json"
        samples = json.loads(source.read_text())
        source_audit[task] = {"source_sha256": digest(source), "selected": ids}
        for source_id in ids:
            index = int(source_id.rsplit("_", 1)[1])
            item = samples[index]
            original, question, target = item["input"], item["question"], item["target"]
            assert babi_oracle(FACT.findall(original), question, task) == target
            eligible = candidates(original, question, task, target)
            match, valid = eligible[0]
            facts = FACT.findall(original)
            fact_index = next(
                i
                for i, m in enumerate(FACT.finditer(original))
                if m.start() == match.start()
            )
            background_ids = encode(FACT.sub("", original))
            noise = tokenizer.decode(
                (background_ids * (8000 // len(background_ids) + 2))[:7600]
            )
            text = "\n".join(facts) + "\n\n" + noise
            edit = list(FACT.finditer(text))[fact_index]
            pool = [(edit[0], target), *valid]
            for variant in VARIANTS:
                sid = f"boundary_{source_id}_{variant}"
                actual_question = (
                    paraphrase(question, task)
                    if variant == "paraphrase"
                    else question.strip()
                )
                label_map = (
                    ({"yes": "true", "no": "false"} if task == "qa9" else ROOM_LABELS)
                    if variant == "labels"
                    else {}
                )
                output_rule = "yes or no." if task == "qa9" else "the location."
                if label_map:
                    output_rule = (
                        "the mapped label. For the FINAL answer only use "
                        + ", ".join(
                            f"{key} => {value}" for key, value in label_map.items()
                        )
                        + "."
                    )
                instruction = (
                    pp[task]["instruction"]
                    + "\n\n"
                    + pp[task]["examples"]
                    + "\n\nFor the final question, give only one answer word: "
                    + output_rule
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
                            + f"End record key: {sid}\nQuestion: {actual_question}",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                offset = rendered.index(text)
                a, c = (
                    encode(rendered[: offset + edit.start()]),
                    encode(rendered[offset + edit.end() :]),
                )
                replacements = [pool[v % len(pool)][0] for v in range(4)]
                bs = [
                    encode(r + f" [record {sid} revision {v:04d}]\n")
                    for v, r in enumerate(replacements)
                ]
                width = max(map(len, bs))
                padding = encode("\n")
                assert len(padding) == 1
                versions = []
                for v, (replacement, b) in enumerate(
                    zip(replacements, bs, strict=True)
                ):
                    value = babi_oracle(
                        FACT.findall(
                            text[: edit.start()] + replacement + text[edit.end() :]
                        ),
                        question,
                        task,
                    )
                    assert value == pool[v % len(pool)][1]
                    b += padding * (width - len(b))
                    versions.append(
                        {
                            "token_ids": a + sep + b + sep + c,
                            "new_chunk_ids": b,
                            "expected_value": label_map.get(value, value),
                            "canonical_value": value,
                            "definition": replacement,
                        }
                    )
                assert len({len(v["token_ids"]) for v in versions}) == 1
                sessions.append(
                    {
                        "session_id": sid,
                        "source_id": source_id,
                        "family": "babilong_" + task,
                        "placement": "early8k",
                        "variant": variant,
                        "prompt_tokens": len(versions[0]["token_ids"]),
                        "editable_token_span": [len(a), len(a) + width],
                        "suffix_tokens": len(c),
                        "versions": versions,
                        "metadata": {
                            "source_index": index,
                            "question": actual_question,
                            "oracle_question": question,
                            "label_map": label_map,
                            "original_facts": facts,
                            "edited_fact_index": fact_index,
                            "previously_evaluated_source": True,
                        },
                    }
                )
    data = {
        "model": model,
        "sessions": sessions,
        "warmup_seed_ids": prior["warmup_seed_ids"],
        "warmup_edit_ids": prior["warmup_edit_ids"],
        "pilot_provenance": prior["pilot_provenance"],
        "source_audit": source_audit,
        "protocol": {
            "stage": "exploratory",
            "previously_evaluated_sources": True,
            "selection": (
                "first sorted source IDs in prior query_fair per family; "
                "no output filtering"
            ),
            "counts": COUNTS,
            "variants": VARIANTS,
            "edits": 3,
            "prior_input_sha256": digest(prior_path),
            "novel_qa9_reserved": [92, 94, 98, 99],
            "caveat": (
                "artificial early8k; adapted public tasks; "
                "not an official benchmark score"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    args.output.with_suffix(".lock.json").write_text(
        json.dumps(
            {
                "input_sha256": digest(args.output),
                "protocol": data["protocol"],
                "builder_sha256": digest(Path(__file__)),
                "source_audit": source_audit,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "sessions": len(sessions),
                "sources": sum(COUNTS.values()),
                "events": len(sessions) * 3,
                "max_tokens": max(s["prompt_tokens"] for s in sessions),
            }
        )
    )


if __name__ == "__main__":
    main()
