import json

from dev.build_ptcore_dataset import adapt_alba, adapt_openbookqa, adapt_pt_exams, adapt_saudade
from nanochat.ptcore_eval import PTCORE_TASKS, random_baseline_for


def test_ptcore_has_six_tasks():
    assert [task["name"] for task in PTCORE_TASKS] == [
        "sst2_pt_mini",
        "alba_mcq",
        "cultura_viva_pt_mcq",
        "pt_exams_history_geography",
        "saudade_pt",
        "openbookqa_mt_pt",
    ]


def test_saudade_uses_answer_text_choices():
    row = adapt_saudade(
        row={
            "prompt": "Qual ocorreu primeiro:",
            "facts_pair": ["Evento recente", "Evento antigo"],
            "answer_index": 1,
            "entity": "Entidade",
            "category": "História",
        },
        index=0,
        source="amalia-llm/saudade-pt",
        config="default",
        split="test",
    )
    assert row["question"] == "Qual ocorreu primeiro:"
    assert row["choices"] == ["Evento recente", "Evento antigo"]
    assert row["correct_choice"] == 1


def test_openbookqa_uses_answer_text_choices():
    row = adapt_openbookqa(
        row={
            "id": "example",
            "question_stem": "A água congela a",
            "choices": {"text": ["0 °C", "50 °C"], "label": ["A", "B"]},
            "answerKey": "A",
        },
        index=0,
        source="amalia-llm/openbookqa-mt-pt",
        config="main",
        split="test",
    )
    assert row["question"] == "A água congela a"
    assert row["choices"] == ["0 °C", "50 °C"]
    assert row["correct_choice"] == 0


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
