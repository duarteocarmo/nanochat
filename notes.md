# Notes

## 2026-05-27: Portuguese nanochat setup

Last commit: `b7a1177 Add Portuguese nanochat setup notes`

- Added `AGENTS.md` with fork-specific guidance: this repo focuses on a budget European Portuguese nanochat adaptation and commits should include short change notes.
- Added Bagaço2 support to `dev/repackage_data_reference.py`, keeping only `text`, `educational_score`, and `category` in repackaged shards.
- Added `dev/estimate_bagaco2_tokens.py` to sample Bagaço2 parquet files and estimate total GPT-2 tokens from total row count.
- Current sampled estimate: ~33.68B GPT-2 tokens for 33,137,796 rows.
- Added `runs/runptcpu.sh` as an initial CPU/MPS scaffold for the Portuguese end-to-end run.
- Added the European Portuguese project TODO list to `README.md`.
- Added a short entry in `dev/LOG.md` for the Bagaço2/PT setup work.

## 2026-05-28: Portuguese tokenizer and CPU run updates

- Pointed the base dataset loader at the Bagaço2 shuffled shards.
- Enabled tokenizer training/eval and tiny PT pretraining in `runs/runptcpu.sh`, with CORE disabled for training and base eval.
- Replaced base-model sample prompts with 7 general European Portuguese prompts.
- Updated README TODOs for completed tokenizer work and Portuguese sample prompts.

## 2026-06-03: Magpie sampling support

- Added deterministic `--sample-size` / `--sample-seed` support to `dev/translate_dataset.py` for translating a limited SmolTalk2 Magpie subset instead of the full split.
- Sampled rows are fetched through the Hugging Face datasets-server rows API, avoiding a full local dataset download before translation starts.
- Added tests for deterministic sampling, paged row fetching, resume, and `--limit` interaction.

## 2026-06-05: Translation context fallback

- Added a context-length fallback to `dev/translate_dataset.py` that catches vLLM/OpenAI-compatible 400 errors, splits only the overlong message text on natural boundaries, retries chunks recursively, and rejoins translated chunks.
- Added tests for context-length error detection, natural-boundary splitting, and split-and-retry translation behavior.

## 2026-06-08: d8 PT training script

- Updated `runs/runptd8.sh` defaults for single-GPU use and ~1,747.6M-token d8 training.

## 2026-06-09: Chinchilla d8/d10 PTCORE findings

- Added a basic Portugal QA PTCORE dataset on Hugging Face: `duarteocarmo/portugal-basic-qa-ptcore`.
- Added `portugal-basic-qa-pt` to PTCORE and temporarily disabled `mmlu-pt` / `goldenswag-pt` to focus on cheaper PT signals.
- Changed Portugal QA scoring from letter completions (`Resposta: b`) to answer-text completions (`Resposta: Lisboa`). This fixed the random/fixed-letter behavior seen in d8.
- d8 Chinchilla run on RTX 5090: 125.8M params, 838.9M tokens, ratio ~20, final val bpb 0.9614, PTCORE 0.1023 with old Portugal QA scoring. Logs are in `logs/chinchillad8/`.
- d10 Chinchilla run on RTX 5090: 196.0M params, 1.402B tokens, ratio ~20, final val bpb 0.9015, PTCORE 0.3740. Logs are in `logs/chinchillad10/`.
- d10 final PTCORE task scores: `sst2-pt` 71.1% raw / 0.4219 centered, `scala-pt` 50.0% / 0.0 centered, `portugal-basic-qa-pt` 80.0% / 0.7000 centered.
- `scala-pt` appeared too hard or poorly prompted for base-model 0-shot eval; reformatted it to explicitly ask whether a quoted sentence is grammatically correct or incorrect.
- d10 validation bpb improved through the final eval, so the model was not saturated. Consider d10 ratio 30 before jumping to d12.
- Final `base_eval` OOMed during BPB with `device_batch_size=32`; reduce final eval batch size to 8 or 4 in future d10+ scripts.
- Removed duplicate wandb logging of generic `centered_results`; use namespaced `ptcore_centered_results` / `core_centered_results` only.
- Added `duarteocarmo/PT-Culture_Data` as an SFT task, converting `human`/`gpt` conversations to `user`/`assistant` messages and splitting train/test by `_seed_id` to avoid fact leakage.
- Renamed the tiny local smoke script to `runs/runptlocaltest.sh` and made it the first gate for new PT pipeline steps: tiny tokenizer, base train/eval, SFT, chat CLI, report generation, with future speedrun steps commented out.
- Added `PT-PortugalBasicQA` as the first Portuguese `chat_eval` task and wired a tiny smoke eval into `runs/runptlocaltest.sh`.
- Renamed Portuguese task files/classes to use the `pt_` / `PT` convention, e.g. `tasks/pt_smoltalk2.py`, `PTSmolTalk2`, and `tasks/pt_portugal_basic_qa_chat.py`.
