"""Portuguese culture SFT dataset."""

import random

from datasets import load_dataset
from tasks.common import Task


DATASET_ID = "duarteocarmo/PT-Culture_Data"
DEFAULT_TEST_SIZE = 0.05
DEFAULT_SPLIT_SEED = 42
ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
}


def split_seed_ids(seed_ids: list[str], test_size: float, split_seed: int) -> tuple[set[str], set[str]]:
    unique_seed_ids = sorted(set(seed_ids))
    rng = random.Random(split_seed)
    rng.shuffle(unique_seed_ids)
    test_count = max(1, round(len(unique_seed_ids) * test_size))
    test_seed_ids = set(unique_seed_ids[:test_count])
    train_seed_ids = set(unique_seed_ids[test_count:])
    return train_seed_ids, test_seed_ids


class PTCulture(Task):
    """PT culture SFT rows converted to the SmolTalk messages format."""

    def __init__(
        self,
        split: str,
        test_size: float = DEFAULT_TEST_SIZE,
        split_seed: int = DEFAULT_SPLIT_SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "PTCulture split must be train|test"
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        self.split = split
        self.ds = load_dataset(path=DATASET_ID, split="train").shuffle(seed=split_seed)
        source_roles = {
            message["role"]
            for row in self.ds
            for message in row["conversations"]
        }
        assert source_roles == set(ROLE_MAP), f"Unexpected PTCulture roles: {sorted(source_roles)}"
        seed_ids = self.ds["_seed_id"]
        train_seed_ids, test_seed_ids = split_seed_ids(
            seed_ids=seed_ids,
            test_size=test_size,
            split_seed=split_seed,
        )
        selected_seed_ids = train_seed_ids if split == "train" else test_seed_ids
        self.indices = [
            index
            for index, seed_id in enumerate(seed_ids)
            if seed_id in selected_seed_ids
        ]
        self.length = len(self.indices)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[self.indices[index]]
        messages = []
        for message in row["conversations"]:
            source_role = message["role"]
            assert source_role in ROLE_MAP, f"Unknown PTCulture role: {source_role}"
            content = message["content"]
            assert isinstance(content, str), "Content must be a string"
            messages.append({"role": ROLE_MAP[source_role], "content": content})

        if messages[-1]["role"] == "user":
            messages = messages[:-1]

        assert len(messages) >= 2, "PTCulture messages must have at least 2 messages"
        assert messages[-1]["role"] == "assistant", "PTCulture messages must end with an assistant response"
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

        return {"messages": messages}
