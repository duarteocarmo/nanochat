"""European Portuguese SmolTalk2 SFT datasets."""

from datasets import load_dataset
from tasks.common import Task


DATASET_ID = "duarteocarmo/smoltalk2PT"
SUBSETS = (
    "everyday_conversations",
    "magpie_ultra",
    "tulu_personas",
    "smol_rewrite",
)


class SmolTalk2PT(Task):
    """Translated SmolTalk2 PT-PT dataset with train/test splits."""

    def __init__(self, subset: str, split: str, **kwargs):
        super().__init__(**kwargs)
        assert subset in SUBSETS, f"SmolTalk2PT subset must be one of {SUBSETS}"
        assert split in ["train", "test"], "SmolTalk2PT split must be train|test"
        self.subset = subset
        self.split = split
        self.ds = load_dataset(DATASET_ID, subset, split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = [dict(message) for message in row["messages"]]
        custom_instructions = row.get("chat_template_kwargs", {}).get("custom_instructions", "")
        if isinstance(custom_instructions, str) and custom_instructions.strip():
            messages = [{"role": "system", "content": custom_instructions.strip()}, *messages]

        assert len(messages) >= 2, "SmolTalk2PT messages must have at least 2 messages"
        if messages[0]["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

        return {"messages": messages}
