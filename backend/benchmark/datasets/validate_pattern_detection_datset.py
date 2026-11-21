
import json
from collections import Counter

with open("pattern_benchmark.json", encoding="utf-8") as f:
    dataset = json.load(f)

assert len(dataset["cases"]) == 50

scenario_counts = Counter(
    case["evaluation"]["scenario"]
    for case in dataset["cases"]
)

expected_distribution = {
    "new_pattern": 15,
    "no_pattern": 10,
    "existing_pattern_update": 10,
    "pattern_resolution": 7,
    "pattern_reactivation": 5,
    "semantic_merge": 3
}

assert scenario_counts == expected_distribution

for case in dataset["cases"]:
    inp = case["input"]
    current = inp["current_episode"]
    episodes = {
        current["episode_id"]: current["summary"]
    }

    for ep in inp["previous_episodes"]:
        assert ep["status"] == "CLOSED"
        assert ep["episode_id"] not in episodes
        episodes[ep["episode_id"]] = ep["summary"]

    existing_ids = {
        p["pattern_id"] for p in inp["existing_patterns"]
    }

    for pattern in case["expected"]["patterns"]:
        ids = set()

        for evidence in pattern["evidence"]:
            eid = evidence["episode_id"]
            quote = evidence["quote"]

            assert eid in episodes
            assert 12 <= len(quote) <= 400
            assert quote.casefold() in episodes[eid].casefold()

            ids.add(eid)

        assert len(ids) >= 2

        pid = pattern["pattern_id"]

        if pid is None:
            assert pattern["status"] == "ACTIVE"
        else:
            assert pid in existing_ids

        if pattern["status"] == "ACTIVE":
            assert current["episode_id"] in ids
            assert pattern["resolution_evidence"] is None

        elif pattern["status"] == "RESOLVED":
            assert pid is not None
            resolution = pattern["resolution_evidence"]
            assert resolution["episode_id"] == current["episode_id"]
            assert resolution["quote"].casefold() in (
                episodes[current["episode_id"]].casefold()
            )

print("Dataset validation passed.")
print(scenario_counts)
