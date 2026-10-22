# SPDX-License-Identifier: Apache-2.0
# Adapted from the existing suffix_public_data.py; source labels audited below.
import re

ROOMS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")

FACT = re.compile(
    r"\b(?:Mary|John|Daniel|Sandra) "
    r"(?:is no longer in|is not in|is in|journeyed to|travelled to|moved to|"
    r"went back to|went to|"
    r"got|grabbed|took|picked up|put down|dropped|discarded|left) "
    r"(?:the )?(?:bathroom|bedroom|garden|hallway|kitchen|office|football|milk|apple)"
    r"(?: there)?\."
)


def babi_oracle(facts, question, task):
    """Replay movements/ownership; history records distinct known locations."""
    people, owners, positions, history = {}, {}, {}, {}
    negative = {}

    def place(obj, room):
        if room is None:
            return
        if positions.get(obj) != room:
            history.setdefault(obj, []).append(room)
        positions[obj] = room

    for fact in facts:
        person = fact.split()[0]
        room = next((r for r in ROOMS if f"the {r}" in fact), None)
        if room:
            if "is no longer in" in fact or "is not in" in fact:
                negative[person] = room
                people[person] = None
            else:
                people[person] = room
                negative.pop(person, None)
                for obj, owner in owners.items():
                    if owner == person:
                        place(obj, room)
        else:
            obj = re.search(r"\b(football|milk|apple)\b", fact)[1]
            if any(v in fact for v in ("put down", "dropped", "discarded", "left")):
                place(obj, people.get(person))
                owners[obj] = None
            else:
                owners[obj] = person
                place(obj, people.get(person))
    if task == "qa9":
        person, room = re.search(r"Is (\w+) in the (\w+)\?", question).groups()
        if people.get(person) is not None:
            return "yes" if people[person] == room else "no"
        return "no" if negative.get(person) == room else "unknown"
    obj = re.search(r"the (football|milk|apple)", question)[1]
    if task == "qa2":
        return positions.get(obj)
    room = re.search(r"before the (\w+)", question)[1]
    trail = history.get(obj, [])
    indices = [i for i, value in enumerate(trail) if value == room]
    return trail[indices[-1] - 1] if indices and indices[-1] > 0 else None
