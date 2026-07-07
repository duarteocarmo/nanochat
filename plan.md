# Dataset Plan

## P1 tokenizer / pretraining
- `duarteocarmo/bagaco2-30b-shuffle` as the main corpus.
- `amalia-llm/CorEGe-PT` as a small continued-pretrain / mix experiment after baseline.

## P1 pretraining validation
- BPB on held-out `bagaco2-30b-shuffle` val shard.
- Optional BPB sanity slice from `CorEGe-PT` if we add it.

## P1 PTCORE
- Keep: `duarteocarmo/sst2-pt-mini`
- Keep: `duarteocarmo/scala-pt`
- Keep: `duarteocarmo/portugal-basic-qa-ptcore`
- Add: `amalia-llm/alba_mcq`
- Add: `amalia-llm/cultura-viva-pt-mcq`
- Add: `amalia-llm/pt_exams`
- Add: `amalia-llm/piqa-mt-pt`

## P1 SFT
- `duarteocarmo/smoltalk2PT`: `everyday_conversations`, `tulu_personas`, `smol_rewrite`, `magpie_ultra`
- `duarteocarmo/PT-Culture_Data`
- `amalia-llm/persona_instruction_following` filtered PT split
- `amalia-llm/persona_nemotron` general + instruction-following
- `amalia-llm/wikipedia_conversations`
- `amalia-llm/ptpt-linguistics-if`

## P1 SFT validation / ChatCORE
- SFT BPB on held-out/test splits from the SFT mix.
- `duarteocarmo/portugal-basic-qa-ptcore`
- hardcoded Portugal city → region QA
- `amalia-llm/alba_mcq`
- `amalia-llm/cultura-viva-pt-mcq`
- `amalia-llm/pt_exams`
- `amalia-llm/P3B3`
- `amalia-llm/xstest_ptpt`
