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
