"""European Portuguese multiple-choice SFT tasks."""

import random

from tasks.common import Task, load_hub_dataset
from tasks.pt_common import render_portuguese_mc


_REPO_ID = "duarteocarmo/pt-mcq-sft"
_SUBSETS = {"mmlu", "goldenswag", "boolq"}


class PTMCQ(Task):
    def __init__(self, subset, split, permutation_seed=0, rows=None, **kwargs):
        super().__init__(**kwargs)
        if subset not in _SUBSETS:
            raise ValueError(f"Unknown PTMCQ subset: {subset}")
        if split != "train":
            raise ValueError(f"PTMCQ split must be train, got {split}")
        self.subset = subset
        self.permutation_seed = permutation_seed
        self.ds = list(rows) if rows is not None else load_hub_dataset(
            repo_id=_REPO_ID,
            subset=subset,
            split=split,
        ).shuffle(seed=42)
        self.length = len(self.ds)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        source_choices = row["choices"]
        permutation = list(range(len(source_choices)))
        random.Random(x=42 + index).shuffle(permutation)
        rotation = self.permutation_seed % len(permutation)
        permutation = permutation[rotation:] + permutation[:rotation]

        choices = [source_choices[source_index] for source_index in permutation]
        correct_choice = permutation.index(row["answer"])
        letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)])
        question = row["question"]
        if row["context"]:
            question = f'Texto: {row["context"]}\n\n{question}'

        return {
            "messages": [
                {
                    "role": "user",
                    "content": render_portuguese_mc(
                        question=question,
                        letters=letters,
                        choices=choices,
                    ),
                },
                {"role": "assistant", "content": letters[correct_choice]},
            ],
            "letters": letters,
            "source": self.subset,
            "source_id": row["source_id"],
        }

    def evaluate(self, conversation, assistant_response):
        return assistant_response == conversation["messages"][-1]["content"]
