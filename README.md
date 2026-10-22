# Whetherman

**A 270M model that forecasts whether your AI agent actually needs to call a tool.**

Agents call tools they don't need. On When2Tool training tasks that a large hosted model (gpt-oss-120b)
reliably answers without tools, it still called a tool on 69 of 161; a small local model (Qwen3.5-4B) did so
on 36 of 68.
Each unneeded call costs a model round trip, tokens and wall time.

Whetherman is a small OpenAI-compatible proxy that sits between an agent and its model. Before each model
call that offers tools, a fine-tuned FunctionGemma-270M forecasts one of four decisions:

| Decision | What the proxy does |
|---|---|
| **ACT** | Forward the request unchanged. This is the default. |
| **ANSWER** | Remove the tools and ask the model to answer from the conversation. |
| **ASK** | Remove the tools and ask the model for one clarifying question. |
| **DECLINE** | Remove the tools and ask the model to explain what it cannot do. |

It only intervenes when a non-ACT decision is confident enough for *that* agent. Anything else, including
a slow or failing classifier, falls back to ACT.

<!-- RESULTS: filled in from the frozen final runs (eval/results/), see below -->

## Why per-agent calibration

Whether a tool is *needed* depends on the agent. "Compute (372*79)-115+13" is something gpt-oss-120b
answers correctly unaided; Qwen3.5-4B does not. A single shared threshold either skips tools the small model
needs or saves nothing on the large one. When the profile fitted for one agent was applied to the other,
false skips went up sharply (see the profile-swap rows in the results).

So Whetherman has two parts:

1. **A shared backbone** (FunctionGemma-270M + LoRA + a 4-class head, no agent identity in the input),
   trained on When2Call, multi-step decision points, and When2Tool tasks where every measured agent agrees.
2. **A per-agent profile**, fitted by `whetherman calibrate` from a probe pack run against *your* agent:
   each probe task is run 3–5 times with and without tools, and the agent's own success decides the label
   (ANSWER if it reliably succeeds unaided, ACT if it needs the tool, otherwise "ambiguous" and unused).
   The profile is a small regularized logistic layer on the backbone's last hidden state plus a threshold τ,
   chosen on a separate split as the smallest τ whose false-skip rate is at most 5%.

## Quick start (from source)

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and access to
[google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) (accept the Gemma
terms on Hugging Face, then put `HF_TOKEN=...` in `.env`). A GPU is optional; the router also runs on CPU.

```bash
git clone https://github.com/vinmay/whetherman && cd whetherman
uv sync --extra ml
```

<!-- BACKBONE: link the released adapter here once published; until then, train it: -->
Train the shared backbone (about 50 minutes on a 6 GB GTX 1660 Ti, FP32):

```bash
uv run python data/prepare.py --download && uv run python data/prepare.py
uv run python -m training.multistep_data
uv run python -m training.train_backbone --out checkpoints/shared-v3 --epochs 3 --soft-disagreement
```

Calibrate your agent (any OpenAI-compatible endpoint). The command prints an estimated cost and asks
before running:

```bash
uv run whetherman calibrate --agent-endpoint http://localhost:11434/v1 --model qwen3.5:4b --name my-agent
# hosted endpoints: --api-key-env NAME_OF_ENV_VAR_WITH_KEY --parallel 6
```

Run the proxy and point your agent at it instead of the model endpoint:

```bash
uv run whetherman serve --profile my-agent --upstream http://localhost:11434/v1 --port 8088
# agent: base_url = http://127.0.0.1:8088/v1
```

Every gated request is logged (`--log decisions.jsonl`) with the decision, P(ACT), τ and router latency.

### Supported subset

The proxy gates non-streaming `POST /v1/chat/completions` requests that include `tools` with
`tool_choice` unset or `"auto"`. Streaming, forced tool choice, requests without tools and every other
endpoint are forwarded unchanged. If a model calls a tool anyway after its tools were removed (gpt-oss does
this), the proxy re-sends the original request, counts it as ACT and records both calls.

## What Whetherman is not

- **Not a security guardrail.** It decides whether a tool call is *useful*, not whether it is *allowed*.
  It fails open by design. Do not use it to block dangerous actions.
- **Not a tool selector.** It does not choose which tool to call or with what arguments.
- **Not free.** A wrong skip can turn a success into a failure. The results report false skips with
  their denominators and confidence intervals.

## Results

<!-- Filled in after the frozen final runs. Every number links to its file in eval/results/. -->

## Limitations

- **Two agents, one day each.** Labels are measured evidence about one agent on one serving stack on one
  day (temperature 0), not ground truth. A different quantization, template or provider is a different agent;
  recalibrate.
- **Small denominators.** The probe packs give each agent about 120 usable labels; held-out sets have tens
  of ACT items per agent. Confidence intervals are wide and are reported.
- **Tool-call non-compliance.** gpt-oss sometimes calls a tool that was removed from the request. The proxy
  falls back to the original request, so those steps save nothing and cost an extra call.
- **Small-agent failures.** On some huge-number arithmetic, Qwen3.5-4B loops until Ollama aborts the
  generation. Those trials count as agent failures (see DATA.md).
- **Guessable tasks.** Two-choice "which city is warmer" tasks can be passed by guessing; they are reported
  separately and excluded from headline success and savings.
- **Supported subset only** (above).

## Reproducing

```bash
uv run python -m eval.run_all --list     # every step, in order
uv run python -m eval.run_all            # development steps (hosted steps need DIGITALOCEAN_INFERENCE_KEY)
```

Held-out evaluations (`ladder_final`, `ab_final_*`, `eval/external/*`) run once, after the backbone,
profiles and code were frozen; any earlier exposure is recorded in [DATA.md](DATA.md). Hosted spend is
logged in `eval/results/spend.json`. Data sources, licenses, splits and labeling rules: [DATA.md](DATA.md).
Models, versions and prices: [MODELS.md](MODELS.md).

## License

Code: Apache-2.0. The backbone is a derivative of Gemma (FunctionGemma-270M) and is subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy. Datasets keep their
own licenses (When2Call CC-BY-4.0, When2Tool MIT); see DATA.md.
