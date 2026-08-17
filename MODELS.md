# Model access metadata

All API results cached in `runs/*/results.jsonl` (raw responses included); analyses are
deterministic from those files (cache-only reproduction: no API keys needed).

| shortname | model ID | provider | access dates (2026) | temp | max_tokens |
|---|---|---|---|---|---|
| gemma27b | google/gemma-3-27b-it | OpenRouter | Aug 13–14 | 1.0 | 6 (60 identity) |
| llama70b | meta-llama/llama-3.1-70b-instruct | OpenRouter | Aug 13–14 | 1.0 | 6/60 |
| gpt41mini | gpt-4.1-mini | OpenAI | Aug 13–14 | 1.0 | 6/60 |
| qwen72b | qwen/qwen-2.5-72b-instruct | OpenRouter | Aug 13 | 1.0 | 6/60 |
| deepseek | deepseek/deepseek-chat-v3-0324 | OpenRouter | Aug 14 | 1.0 | 6/60 |
| mistral | mistralai/mistral-small-3.2-24b-instruct | OpenRouter | Aug 14 | 1.0 | 6/60 |
| kimi | moonshotai/kimi-k2 | OpenRouter | Aug 14 | 1.0 | 6/60 |
| cohere | cohere/command-r-08-2024 | OpenRouter | Aug 14 | 1.0 | 6/60 |
| gemma12b | google/gemma-3-12b-it | OpenRouter | Aug 14 | 1.0 | 6/60 |
| llama8b | meta-llama/llama-3.1-8b-instruct | OpenRouter | Aug 14 | 1.0 | 6/60 |
| llama33 | meta-llama/llama-3.3-70b-instruct | OpenRouter | Aug 14 | 1.0 | 6/60 |
| gpt4omini | gpt-4o-mini | OpenAI | Aug 14 | 1.0 | 6/60 |

GPU stages (activations, exact logits, steering): google/gemma-3-27b-it weights from
Hugging Face, bf16, RunPod A100-SXM4-80GB, transformers eager attention, Aug 13–14.
Derived-artifact SHA-256 checksums: `runs/artifact_checksums.txt`. Hosted models are
mutable; results reflect the access dates above.
