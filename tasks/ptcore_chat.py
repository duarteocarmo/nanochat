"""Portuguese chat evaluation tasks backed by PTCORE."""

import random

from nanochat.ptcore_eval import PTCORE_REPO_ID, default_local_ptcore_dir, load_ptcore_task
from tasks.common import Task
from tasks.pt_common import render_portuguese_mc


PTCORE_CHAT_TASKS = (
    {"name": "PT-PortugalBasicQA", "family": "portugal", "subset": "portugal_basic_qa", "baseline": 1 / 3},
    {"name": "PT-PortugalCityRegion", "family": "portugal", "baseline": 1 / 7},
    {"name": "PT-ALBA-CultureBoundSemantics", "family": "alba", "subset": "alba_mcq_culture_bound_semantics", "baseline": 1 / 3},
    {"name": "PT-ALBA-DiscourseAnalysis", "family": "alba", "subset": "alba_mcq_discourse_analysis", "baseline": 1 / 3},
    {"name": "PT-ALBA-LanguageVariety", "family": "alba", "subset": "alba_mcq_language_variety", "baseline": 1 / 3},
    {"name": "PT-ALBA-Morphology", "family": "alba", "subset": "alba_mcq_morphology", "baseline": 1 / 3},
    {"name": "PT-ALBA-PhoneticsPhonology", "family": "alba", "subset": "alba_mcq_phonetics_phonology", "baseline": 1 / 3},
    {"name": "PT-ALBA-Syntax", "family": "alba", "subset": "alba_mcq_syntax", "baseline": 1 / 3},
    {"name": "PT-ALBA-WordPlay", "family": "alba", "subset": "alba_mcq_word_play", "baseline": 1 / 3},
    {"name": "PT-CulturaViva", "family": "cultura_viva", "subset": "cultura_viva_pt_mcq", "baseline": 1 / 4},
    {"name": "PT-Exams-BioGeo", "family": "pt_exams", "subset": "pt_exams_bio_geo", "baseline": 1 / 4},
    {"name": "PT-Exams-Geography", "family": "pt_exams", "subset": "pt_exams_geography", "baseline": 1 / 4},
    {"name": "PT-Exams-HistoryA", "family": "pt_exams", "subset": "pt_exams_history_a", "baseline": 1 / 4},
    {"name": "PT-Exams-MathematicsA", "family": "pt_exams", "subset": "pt_exams_mathematics_a", "baseline": 1 / 4},
    {"name": "PT-Exams-Philosophy", "family": "pt_exams", "subset": "pt_exams_philosophy", "baseline": 1 / 4},
    {"name": "PT-Exams-Portuguese", "family": "pt_exams", "subset": "pt_exams_portuguese", "baseline": 1 / 4},
)
PTCORE_CHAT_TASK_NAMES = tuple(task["name"] for task in PTCORE_CHAT_TASKS)
PTCORE_CHAT_TASK_BY_NAME = {task["name"]: task for task in PTCORE_CHAT_TASKS}

REGIONS = (
    "Norte",
    "Centro",
    "Área Metropolitana de Lisboa",
    "Alentejo",
    "Algarve",
    "Região Autónoma dos Açores",
    "Região Autónoma da Madeira",
)
CITY_REGIONS = (
    ("Porto", "Norte"),
    ("Braga", "Norte"),
    ("Guimarães", "Norte"),
    ("Viana do Castelo", "Norte"),
    ("Coimbra", "Centro"),
    ("Aveiro", "Centro"),
    ("Viseu", "Centro"),
    ("Leiria", "Centro"),
    ("Lisboa", "Área Metropolitana de Lisboa"),
    ("Sintra", "Área Metropolitana de Lisboa"),
    ("Cascais", "Área Metropolitana de Lisboa"),
    ("Amadora", "Área Metropolitana de Lisboa"),
    ("Évora", "Alentejo"),
    ("Beja", "Alentejo"),
    ("Portalegre", "Alentejo"),
    ("Elvas", "Alentejo"),
    ("Faro", "Algarve"),
    ("Lagos", "Algarve"),
    ("Portimão", "Algarve"),
    ("Albufeira", "Algarve"),
    ("Ponta Delgada", "Região Autónoma dos Açores"),
    ("Angra do Heroísmo", "Região Autónoma dos Açores"),
    ("Horta", "Região Autónoma dos Açores"),
    ("Ribeira Grande", "Região Autónoma dos Açores"),
    ("Funchal", "Região Autónoma da Madeira"),
    ("Santa Cruz", "Região Autónoma da Madeira"),
    ("Câmara de Lobos", "Região Autónoma da Madeira"),
    ("Machico", "Região Autónoma da Madeira"),
)


def conversation_for(question, choices, correct_choice):
    letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)])
    answer = letters[correct_choice]
    return {
        "messages": [
            {"role": "user", "content": render_portuguese_mc(question=question, letters=letters, choices=choices)},
            {"role": "assistant", "content": answer},
        ],
        "letters": letters,
    }


class PTCoreChatMCQ(Task):
    def __init__(self, subset, rows=None, **kwargs):
        super().__init__(**kwargs)
        valid_subsets = {task["subset"] for task in PTCORE_CHAT_TASKS if "subset" in task}
        if subset not in valid_subsets:
            raise ValueError(f"Unknown PTCORE-Chat subset: {subset}")
        self.rows = list(rows) if rows is not None else load_ptcore_task(
            task_name=subset,
            repo_id=PTCORE_REPO_ID,
            local_dir=default_local_ptcore_dir(),
        )
        random.Random(x=42).shuffle(self.rows)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        return conversation_for(
            question=row["query"],
            choices=row["choices"],
            correct_choice=row["gold"],
        )

    def evaluate(self, conversation, assistant_response):
        if assistant_response not in conversation["letters"]:
            raise ValueError(f"Answer {assistant_response} must be one of {conversation['letters']}")
        return assistant_response == conversation["messages"][-1]["content"]


class PTCityRegionQA(Task):
    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(CITY_REGIONS)

    def get_example(self, index):
        city, region = CITY_REGIONS[index]
        choices = list(REGIONS)
        random.Random(x=42 + index).shuffle(choices)
        return conversation_for(
            question=f"A que região de Portugal pertence a cidade de {city}?",
            choices=choices,
            correct_choice=choices.index(region),
        )

    def evaluate(self, conversation, assistant_response):
        if assistant_response not in conversation["letters"]:
            raise ValueError(f"Answer {assistant_response} must be one of {conversation['letters']}")
        return assistant_response == conversation["messages"][-1]["content"]


def build_ptcore_chat_task(task_name):
    task = PTCORE_CHAT_TASK_BY_NAME[task_name]
    if task_name == "PT-PortugalCityRegion":
        return PTCityRegionQA()
    return PTCoreChatMCQ(subset=task["subset"])


def aggregate_ptcore_chat(results):
    centered_by_family = {}
    for family in dict.fromkeys(task["family"] for task in PTCORE_CHAT_TASKS):
        family_tasks = [task for task in PTCORE_CHAT_TASKS if task["family"] == family]
        centered = [
            (results[task["name"]] - task["baseline"]) / (1 - task["baseline"])
            for task in family_tasks
        ]
        centered_by_family[family] = sum(centered) / len(centered)
    metric = sum(centered_by_family.values()) / len(centered_by_family)
    return {"metric": metric, "families": centered_by_family}
