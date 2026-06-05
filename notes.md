# Notes

## 2026-05-27: Portuguese nanochat setup

Last commit: `b7a1177 Add Portuguese nanochat setup notes`

- Added `AGENTS.md` with fork-specific guidance: this repo focuses on a budget European Portuguese nanochat adaptation and commits should include short change notes.
- Added Bagaço2 support to `dev/repackage_data_reference.py`, keeping only `text`, `educational_score`, and `category` in repackaged shards.
- Added `dev/estimate_bagaco2_tokens.py` to sample Bagaço2 parquet files and estimate total GPT-2 tokens from total row count.
- Current sampled estimate: ~33.68B GPT-2 tokens for 33,137,796 rows.
- Added `runs/runcpu-pt.sh` as an initial CPU/MPS scaffold for the Portuguese end-to-end run.
- Added the European Portuguese project TODO list to `README.md`.
- Added a short entry in `dev/LOG.md` for the Bagaço2/PT setup work.

## 2026-05-28: Portuguese tokenizer and CPU run updates

- Pointed the base dataset loader at the Bagaço2 shuffled shards.
- Enabled tokenizer training/eval and tiny PT pretraining in `runs/runcpu-pt.sh`, with CORE disabled for training and base eval.
- Replaced base-model sample prompts with 7 general European Portuguese prompts.
- Updated README TODOs for completed tokenizer work and Portuguese sample prompts.

## 2026-06-03: Magpie sampling support

- Added deterministic `--sample-size` / `--sample-seed` support to `dev/translate_dataset.py` for translating a limited SmolTalk2 Magpie subset instead of the full split.
- Sampled rows are fetched through the Hugging Face datasets-server rows API, avoiding a full local dataset download before translation starts.
- Added tests for deterministic sampling, paged row fetching, resume, and `--limit` interaction.

## 2026-06-05: Translation context fallback

- Added a context-length fallback to `dev/translate_dataset.py` that catches vLLM/OpenAI-compatible 400 errors, splits only the overlong message text on natural boundaries, retries chunks recursively, and rejoins translated chunks.
- Added tests for context-length error detection, natural-boundary splitting, and split-and-retry translation behavior.
