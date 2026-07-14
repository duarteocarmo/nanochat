"""Standardized European Portuguese AMALIA SFT datasets."""

from tasks.common import Task, load_hub_dataset
from tasks.pt_common import normalize_conversation


_SUBSETS = {
    "pt_culture",
    "pt_persona_instruction",
    "pt_nemotron_instruction",
    "pt_nemotron_general",
    "pt_wikipedia",
    "pt_linguistics",
}


class PTAmaliaSFT(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        if subset not in _SUBSETS:
            raise ValueError(f"Unknown PTAmaliaSFT subset: {subset}")
        if split not in {"train", "validation"}:
            raise ValueError(f"PTAmaliaSFT split must be train|validation, got {split}")
        self.ds = load_hub_dataset(
            repo_id="duarteocarmo/amalia-sft",
            subset=subset,
            split=split,
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return normalize_conversation(messages=self.ds[index]["messages"])
