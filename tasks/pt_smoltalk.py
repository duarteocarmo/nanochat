"""European Portuguese SmolTalk conversation subsets."""

from tasks.common import Task, load_hub_dataset
from tasks.pt_common import normalize_conversation


_SUBSETS = {
    "everyday": "everyday_conversations",
    "magpie": "magpie_ultra",
    "rewrite": "smol_rewrite",
    "tulu": "tulu_personas",
}


class PTSmolTalk(Task):
    def __init__(self, subset, split, **kwargs):
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
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return normalize_conversation(messages=self.ds[index]["messages"])
