"""Portugal basic QA chat evaluation task."""

from datasets import load_dataset
from tasks.common import Task


DATASET_ID = "duarteocarmo/portugal-basic-qa-ptcore"
LETTERS = ("A", "B", "C", "D")
LABEL_TO_LETTER = {letter.lower(): letter for letter in LETTERS}


def render_portuguese_mc(question: str, letters: list[str], choices: list[str]) -> str:
    prompt = f"Pergunta de escolha múltipla: {question}\n"
    prompt += "".join(
        f"- {choice}={letter}\n"
        for letter, choice in zip(letters, choices)
    )
    prompt += "\nResponde apenas com a letra da resposta correta."
    return prompt


class PTPortugalBasicQAChat(Task):
    """Simple Portuguese culture/geography QA as chat multiple choice."""

    def __init__(self, split: str = "val", **kwargs):
        super().__init__(**kwargs)
        assert split == "val", "PTPortugalBasicQAChat currently only has a val split"
        self.split = split
        self.ds = load_dataset(path=DATASET_ID, split=split, token=True).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        choices = list(row["choices"])
        letters = list(LETTERS[:len(choices)])
        answer_letter = LABEL_TO_LETTER[row["label"]]
        assert answer_letter in letters, f"Answer {answer_letter} must be one of {letters}"
        messages = [
            {
                "role": "user",
                "content": render_portuguese_mc(
                    question=row["question"],
                    letters=letters,
                    choices=choices,
                ),
            },
            {"role": "assistant", "content": answer_letter},
        ]
        return {
            "messages": messages,
            "letters": letters,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], f"PTPortugalBasicQAChat answer {assistant_response} is expected to be one of {conversation['letters']}"
        answer_letter = conversation["messages"][-1]["content"]
        return assistant_response == answer_letter
