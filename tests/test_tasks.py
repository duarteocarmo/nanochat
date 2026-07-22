"""
Test the Task container machinery: slicing views, mixtures, and the
HubDataset parquet wrapper (in-memory, no network).

python -m pytest tests/test_tasks.py -v
"""

import numpy as np
import pyarrow as pa
from tasks.common import Task, TaskMixture, HubDataset, render_mc
from tasks.pt_common import normalize_conversation, render_portuguese_mc
from tasks.pt_mcq import PTMCQ
from tasks.ptcore_chat import (
    PTCORE_CHAT_TASKS,
    PTCityRegionQA,
    PTCoreChatMCQ,
    aggregate_ptcore_chat,
)


class ToyTask(Task):
    """A trivial task: example i is just {'i': i, 'tag': tag}."""

    def __init__(self, n=10, tag="a", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.tag = tag

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index, "tag": self.tag}


def test_task_full():
    task = ToyTask(n=10)
    assert len(task) == 10
    assert task[0] == {"i": 0, "tag": "a"}
    assert task[9] == {"i": 9, "tag": "a"}


def test_task_slicing():
    # a view of [5, 10) has 5 examples and maps logical to physical indices
    task = ToyTask(n=10, start=5, stop=10)
    assert len(task) == 5
    assert task[0]["i"] == 5
    # step slicing uses ceil division for the length
    task = ToyTask(n=10, start=0, stop=10, step=3) # 0, 3, 6, 9
    assert len(task) == 4
    assert [task[i]["i"] for i in range(4)] == [0, 3, 6, 9]


def test_mixture_covers_all_examples_deterministically():
    mixture = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert len(mixture) == 8
    examples = [mixture[i] for i in range(8)]
    # every example appears exactly once
    keys = sorted((ex["tag"], ex["i"]) for ex in examples)
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    # the shuffle is deterministic: a second instance yields the same order
    mixture2 = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert examples == [mixture2[i] for i in range(8)]
    # and the tasks are actually interleaved, not concatenated
    assert [ex["tag"] for ex in examples] != ["a"] * 3 + ["b"] * 5


def test_mixture_oversampling():
    # passing a task twice doubles its examples
    mixture = TaskMixture([ToyTask(n=3), ToyTask(n=3)])
    assert len(mixture) == 6


def test_hub_dataset_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubDataset(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_dataset_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubDataset(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubDataset(table)[0] == {"x": 0}


def test_normalize_pt_conversation():
    conversation = normalize_conversation(messages=[
        {"from": "human", "value": "Pergunta"},
        {"from": "gpt", "value": "Resposta"},
    ])
    assert conversation["messages"] == [
        {"role": "user", "content": "Pergunta"},
        {"role": "assistant", "content": "Resposta"},
    ]


def test_render_mc_letter_binding():
    query = render_mc("What is 1+1?", ("A", "B"), ("1", "2"))
    # the letter must directly follow '=' with no whitespace, so that the
    # prompt token for "A" matches the assistant's bare "A" response token
    assert "=A\n" in query and "=B\n" in query


def test_render_portuguese_mc_letter_binding():
    query = render_portuguese_mc(
        question="Quanto é 1+1?",
        letters=("A", "B"),
        choices=("1", "2"),
    )
    assert "=A\n" in query and "=B\n" in query
    assert query.endswith("Responde apenas com a letra da resposta correta.")


def test_pt_mcq_permutations_cover_answer_letters():
    rows = [{
        "question": "Quanto é 1+1?",
        "context": "Usa aritmética.",
        "choices": ["0", "1", "2", "3"],
        "answer": 2,
        "source_id": "example-1",
    }]
    conversations = [
        PTMCQ(subset="mmlu", split="train", permutation_seed=seed, rows=rows)[0]
        for seed in range(4)
    ]
    answers = {conversation["messages"][-1]["content"] for conversation in conversations}
    assert answers == {"A", "B", "C", "D"}
    for conversation in conversations:
        answer = conversation["messages"][-1]["content"]
        assert f"- 2={answer}\n" in conversation["messages"][0]["content"]
        assert "Texto: Usa aritmética." in conversation["messages"][0]["content"]


def test_ptcore_chat_mcq_adapter():
    task = PTCoreChatMCQ(
        subset="portugal_basic_qa",
        rows=[{"query": "Qual é a capital?", "choices": ["Porto", "Lisboa", "Faro"], "gold": 1}],
    )
    conversation = task[0]
    assert conversation["letters"] == ("A", "B", "C")
    assert conversation["messages"][-1] == {"role": "assistant", "content": "B"}
    assert task.evaluate(conversation=conversation, assistant_response="B")
    assert not task.evaluate(conversation=conversation, assistant_response="A")


def test_pt_city_region_qa():
    task = PTCityRegionQA()
    assert len(task) == 28
    for index in range(len(task)):
        conversation = task[index]
        assert len(conversation["letters"]) == 7
        assert task.evaluate(
            conversation=conversation,
            assistant_response=conversation["messages"][-1]["content"],
        )


def test_ptcore_chat_aggregation_baseline_and_perfect():
    baseline_results = {task["name"]: task["baseline"] for task in PTCORE_CHAT_TASKS}
    baseline = aggregate_ptcore_chat(results=baseline_results)
    assert baseline["metric"] == 0
    assert set(baseline["families"]) == {"portugal", "alba", "cultura_viva", "pt_exams"}

    perfect_results = {task["name"]: 1.0 for task in PTCORE_CHAT_TASKS}
    perfect = aggregate_ptcore_chat(results=perfect_results)
    assert perfect["metric"] == 1
