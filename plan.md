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
- [x] Standardize PTCORE sources into `duarteocarmo/ptcore`.
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
- [ ] Add `duarteocarmo/smoltalk2PT`: `everyday_conversations`, `tulu_personas`, `smol_rewrite`, `magpie_ultra`.
- [ ] Add `duarteocarmo/PT-Culture_Data`.
- [ ] Add `amalia-llm/persona_instruction_following` filtered PT split.
- [ ] Add `amalia-llm/persona_nemotron` general + instruction-following.
- [ ] Add `amalia-llm/wikipedia_conversations`.
- [ ] Add `amalia-llm/ptpt-linguistics-if`.

## P1 SFT validation / ChatCORE
- [ ] Run SFT BPB on held-out/test splits from the SFT mix.
- [ ] Add `duarteocarmo/portugal-basic-qa-ptcore`.
- [ ] Add hardcoded Portugal city → region QA.
- [ ] Add `amalia-llm/alba_mcq`.
- [ ] Add `amalia-llm/cultura-viva-pt-mcq`.
- [ ] Add `amalia-llm/pt_exams`.
- [ ] Add `amalia-llm/P3B3`.
- [ ] Add `amalia-llm/xstest_ptpt`.
