import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "dev" / "translate_dataset.py"
spec = importlib.util.spec_from_file_location("translate_dataset", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
translate_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translate_dataset)


def args_for(**overrides):
    values = {
        "dataset": "dataset",
        "config": "config",
        "split": "split",
        "limit": None,
        "sample_size": 3,
        "sample_seed": 42,
        "sample_page_size": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sample_indices_are_deterministic_and_not_contiguous():
    first = translate_dataset.sample_indices_for(
        total_rows=20,
        sample_size=5,
        seed=42,
    )
    second = translate_dataset.sample_indices_for(
        total_rows=20,
        sample_size=5,
        seed=42,
    )

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert first != list(range(5))


def test_sample_indices_cap_to_total_rows():
    indices = translate_dataset.sample_indices_for(
        total_rows=3,
        sample_size=10,
        seed=42,
    )

    assert indices == [0, 1, 2]


def test_sampled_rows_are_loaded_by_page_without_full_dataset(monkeypatch):
    calls = []

    def fake_fetch_dataset_rows_page(*, dataset, config, split, offset, length):
        calls.append({"offset": offset, "length": length})
        return {
            "num_rows_total": 20,
            "rows": [
                {"row_idx": index, "row": {"id": index}}
                for index in range(offset, min(offset + length, 20))
            ],
        }

    monkeypatch.setattr(
        translate_dataset,
        "fetch_dataset_rows_page",
        fake_fetch_dataset_rows_page,
    )

    rows = list(translate_dataset.iter_sampled_rows(args=args_for(), start_index=0))

    assert rows == [{"id": 0}, {"id": 3}, {"id": 8}]
    assert calls == [
        {"offset": 0, "length": 1},
        {"offset": 0, "length": 5},
        {"offset": 5, "length": 5},
    ]


def test_sampled_rows_respect_resume_and_limit(monkeypatch):
    def fake_fetch_dataset_rows_page(*, dataset, config, split, offset, length):
        return {
            "num_rows_total": 20,
            "rows": [
                {"row_idx": index, "row": {"id": index}}
                for index in range(offset, min(offset + length, 20))
            ],
        }

    monkeypatch.setattr(
        translate_dataset,
        "fetch_dataset_rows_page",
        fake_fetch_dataset_rows_page,
    )

    rows = list(
        translate_dataset.iter_sampled_rows(
            args=args_for(sample_size=5, limit=3),
            start_index=1,
        )
    )

    assert rows == [{"id": 3}, {"id": 8}]


def test_translate_text_does_not_limit_output_tokens():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "olá"}}]}

    class FakeClient:
        def __init__(self):
            self.payload = None

        def post(self, *, url, json):
            self.payload = json
            return FakeResponse()

    client = FakeClient()

    translated = translate_dataset.translate_text(
        client=client,
        endpoint="http://localhost:18000/v1/chat/completions",
        model="translategemma-12b-it",
        text="hello",
        source="en",
        target="pt-PT",
        temperature=0.0,
        retries=1,
    )

    assert translated == "olá"
    assert client.payload is not None
    assert "max_tokens" not in client.payload
