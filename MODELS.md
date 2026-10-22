# Models

All measured models are open-weight. Rows marked *unverified* are re-checked against the provider
before any large run and updated here.

| Role | Model | Where | License | Verified |
|---|---|---|---|---|
| Whetherman backbone | `google/functiongemma-270m-it`, revision `39eccb091651513a5dfb56892d3714c1b5b8276c` | local (GTX 1660 Ti, FP32) or CPU | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (gated; accepted by the owner) | 2026-10-21 |
| Agent S | Qwen3.5-4B, Ollama `qwen3.5:4b` (Q4_K_M, digest `d8b0f5e9760c`), thinking disabled via `reasoning_effort: none` | local Ollama 0.40.4 | Apache-2.0 | 2026-10-21 |
| Agent L | `openai-gpt-oss-120b` (gpt-oss-120b) | DigitalOcean serverless inference | Apache-2.0 | 2026-10-21 (listed by `/v1/models`; sequential tool-calling test passes) |
| Agent U (P1, unseen) | `openai-gpt-oss-20b` | DigitalOcean serverless inference | Apache-2.0 | 2026-10-22 (calibrated with `whetherman calibrate` as a new agent) |
| Prompted router baseline | Qwen3.5-4B (same as Agent S) | local Ollama | Apache-2.0 | 2026-10-21 |
| Dedup embeddings | `embeddinggemma:300m` (digest `85462619ee72`) | local Ollama | Gemma Terms of Use | 2026-10-21 |

## Prices used in cost accounting

`whetherman/providers/openai_compat.py::PROVIDERS` holds per-model prices (USD per 1M tokens). The
DigitalOcean values were checked against DigitalOcean's pricing page on 2026-10-21: `gpt-oss-120b` $0.10 input /
$0.70 output, `gpt-oss-20b` $0.05 / $0.45. No cached-input price is listed, so cached tokens are billed as input.
Local Ollama is counted at $0. Hosted spend is capped at $10 and logged in `eval/results/spend.json`.

## Agent S tool-calling check

Ollama 0.40.4 parses Qwen3.5 tool calls through the OpenAI-compatible endpoint. A sequential test
(get_weather → calculator → answer) round-tripped tool-call ids and results correctly, and the endpoint
reports cached prompt tokens (`prompt_tokens_details.cached_tokens`).

Ollama aborts some degenerate Qwen3.5-4B generations (digit loops on huge-number arithmetic) with HTTP 500
`prediction aborted, token repeat limit reached`. The client treats that response as the agent failing the turn
and does not retry it; other 5xx responses are retried as transient (DATA.md).
