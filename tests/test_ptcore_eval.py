import json

from dev.build_ptcore_dataset import adapt_alba, adapt_pt_exams
from nanochat.ptcore_eval import PTCORE_TASKS, random_baseline_for


def test_ptcore_has_five_tasks():
    assert [task["name"] for task in PTCORE_TASKS] == [
        "sst2_pt_mini",
        "portugal_basic_qa",
        "alba_mcq",
        "cultura_viva_pt_mcq",
        "pt_exams_history_geography",
    ]


def test_alba_domains_are_pooled():
    row = adapt_alba(
        row={
            "id": "example",
            "question": "Pergunta?",
            "choices": ["A", "B", "C"],
            "correct_choice": 1,
            "subject": "Lexicology",
            "scores": None,
        },
        index=0,
        source="amalia-llm/alba_mcq",
        config="lexicology",
        split="test",
    )
    assert row["task"] == "alba_mcq"
    assert row["source_config"] == "lexicology"


def test_pt_exams_pool_history_and_geography_only():
    base_row = {
        "year": 2020,
        "phase": "1",
        "question": "Pergunta?",
        "choices": ["A", "B", "C", "D"],
        "answer": 2,
        "question_group": "I",
        "question_number": "1",
        "is_completion": False,
    }
    geography = adapt_pt_exams(
        row={**base_row, "subject": "Geography"},
        index=0,
        source="amalia-llm/pt_exams",
        config="default",
        split="test",
    )
    history = adapt_pt_exams(
        row={**base_row, "subject": "History A"},
        index=1,
        source="amalia-llm/pt_exams",
        config="default",
        split="test",
    )
    mathematics = adapt_pt_exams(
        row={**base_row, "subject": "Mathematics A"},
        index=2,
        source="amalia-llm/pt_exams",
        config="default",
        split="test",
    )

    assert geography["task"] == "pt_exams_history_geography"
    assert history["task"] == "pt_exams_history_geography"
    assert json.loads(geography["metadata"])["subject"] == "Geography"
    assert mathematics is None


def test_random_baseline_supports_mixed_choice_counts():
    data = [
        {"choices": ["A", "B"]},
        {"choices": ["A", "B", "C", "D"]},
    ]
    assert random_baseline_for(data=data) == 0.375
