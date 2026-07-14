# Dataset Plan Checklist

## P1 tokenizer / pretraining
- [x] Use `duarteocarmo/bagaco2-30b-shuffle` as the main corpus.
- [x] Confirm Bagaço2 format matches nanochat pretraining loader.
- [x] Point tokenizer/pretraining data loader to Bagaço2.
- [x] Update tokenizer eval labels to Bagaço2.
- [ ] Add `amalia-llm/CorEGe-PT` as a small continued-pretrain / mix experiment after baseline.

## P1 pretraining validation
- [x] Run BPB on held-out `bagaco2-30b-shuffle` val shard.
- [ ] Add optional BPB sanity slice from `CorEGe-PT` if we add it.

## P1 PTCORE
- [x] Standardize PTCORE sources into `duarteocarmo/ptcore-eval`.
- [x] Use one HF config/subset per PTCORE task.
- [x] Keep `duarteocarmo/sst2-pt-mini`.
- [x] Keep `duarteocarmo/scala-pt`.
- [x] Keep `duarteocarmo/portugal-basic-qa-ptcore`.
- [x] Add `amalia-llm/alba_mcq`.
- [x] Add `amalia-llm/cultura-viva-pt-mcq`.
- [x] Add `amalia-llm/pt_exams`.
- [x] Add `amalia-llm/piqa-mt-pt`.
- [x] Replace base `core` eval with PTCORE.
- [x] Support per-task few-shot config for future PTCORE variants.
- [x] Print selected/available example counts during PTCORE eval.

## P1 mini integration run
- [x] Add a tiny local speedrun for Mac/CPU/MPS.
- [x] Validate tokenizer → base train → PTCORE/base eval → SFT → chat eval flow.
- [x] Run full PTCORE by default in tiny speedrun.

## P1 SFT
- [x] Add `duarteocarmo/smoltalk2PT`: `everyday_conversations`, `tulu_personas`, `smol_rewrite`, `magpie_ultra`.
- [x] Add `amalia-llm/PT-Culture_Data`.
- [x] Add `amalia-llm/persona_instruction_following` filtered PT split.
- [x] Add `amalia-llm/persona_nemotron` general + instruction-following.
- [x] Add `amalia-llm/wikipedia_conversations`.
- [x] Add `amalia-llm/ptpt-linguistics-if` at 10x weight.
- [x] Consolidate AMALIA sources into `duarteocarmo/amalia-sft` with standardized Parquet configs and splits.
- [x] Use dedicated SmolTalk2PT test splits and deterministic 2% holdouts for the other sources.

## P1 SFT validation / ChatCORE
- [ ] Run SFT BPB on held-out/test splits from the SFT mix.
- [ ] Add `duarteocarmo/portugal-basic-qa-ptcore`.
- [ ] Add hardcoded Portugal city → region QA.
- [ ] Add `amalia-llm/alba_mcq`.
- [ ] Add `amalia-llm/cultura-viva-pt-mcq`.
- [ ] Add `amalia-llm/pt_exams`.
- [ ] Add `amalia-llm/P3B3`.
- [ ] Add `amalia-llm/xstest_ptpt`.

# Issues
- [ ] Fix SFT stopping so `--num-iterations=N` trains exactly N updates instead of stopping on the prefetched batch.
- [ ] Fix one-epoch DDP stopping so training does not end when only the first rank exhausts its conversations.
