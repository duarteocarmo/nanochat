"""European Portuguese SmolTalk conversation subsets."""

from tasks.common import Task, load_hub_dataset
from tasks.pt_common import conversation_indices, normalize_conversation


_SUBSETS = {
    "everyday": "everyday_conversations",
    "magpie": "magpie_ultra",
    "rewrite": "smol_rewrite",
    "tulu": "tulu_personas",
}


class PTSmolTalk(Task):
    def __init__(self, subset, split, tokenizer=None, max_assistant_tokens=-1, **kwargs):
        super().__init__(**kwargs)
        if subset not in _SUBSETS:
            raise ValueError(f"Unknown PTSmolTalk subset: {subset}")
        if split not in {"train", "validation"}:
            raise ValueError(f"PTSmolTalk split must be train|validation, got {split}")
        hub_split = "train" if split == "train" else "test"
        self.ds = load_hub_dataset(
            repo_id="duarteocarmo/smoltalk2PT",
            subset=_SUBSETS[subset],
            split=hub_split,
        ).shuffle(seed=42)
        if max_assistant_tokens >= 0 and tokenizer is None:
            raise ValueError("tokenizer is required with max_assistant_tokens")
        self.indices = conversation_indices(
            dataset=self.ds,
            tokenizer=tokenizer,
            max_assistant_tokens=max_assistant_tokens,
            stop=self.stop,
        )
        self.length = len(self.ds) if self.indices is None else len(self.indices)
        if self.stop is not None:
            self.stop = min(self.stop, self.length)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        physical_index = index if self.indices is None else self.indices[index]
        return normalize_conversation(messages=self.ds[physical_index]["messages"])
