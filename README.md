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

**The short version of the results:** on held-out tasks, Whetherman did not make any of the three agents
better. It was roughly neutral on a small local model and on a gpt-oss-20b agent it had never seen
(−2 and −1 successes of 150, with 1% and 8% fewer tool calls), and it hurt the large gpt-oss-120b agent
(−15 of 150, with higher cost). Its clearest win is offline: per-agent calibration cuts false skips from
65/70 to 4/70 on the small agent. [Full results below.](#results)

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

Everything below was measured on held-out sets, run once after the backbone, profiles and code were frozen
(`eval/results/freeze.json`, tag `freeze-v3`). Full tables with every configuration, CI and set:
[eval/results/REPORT_final.md](eval/results/REPORT_final.md), generated by `eval/report.py` from the raw
result files.

### End to end: no gate vs Whetherman

80 held-out tasks (40 multi-step) × 2 repeats per agent. "Headline" excludes 10 guessable two-choice episodes
(DATA.md). CIs are Wilson 95%.

| Agent | No gate | Whetherman | Tool calls / episode | Tokens / episode | Cost (whole suite) |
|---|---|---|---|---|---|
| L: gpt-oss-120b (DigitalOcean) | 120/150 [0.73, 0.86] | **105/150** [0.62, 0.77] | 2.36 → 2.15 | 1906 → 1897 | $0.0522 → $0.0651 |
| S: Qwen3.5-4B (local Ollama) | 130/150 [0.80, 0.91] | **128/150** [0.79, 0.90] | 1.92 → 1.91 | 2297 → 2156 | local |
| U: gpt-oss-20b (unseen agent) | 115/150 [0.69, 0.83] | **114/150** [0.69, 0.82] | 2.09 → 1.93 | 2062 → 1824 | $0.0461 → $0.0438 |

Other gates in the same runs (headline success): the shared backbone without a profile 99 (L) and 84 (S);
a prompted Qwen3.5-4B router 108 and 96; a rules-only control 90 and 60. Every gate that removed many calls
lost many successes.

**What went wrong for L** (post-hoc analysis of the frozen run, not used to change anything):
- 11 of the 15 lost successes have replay evidence of a false skip: restoring ACT at the skipped step made
  the episode succeed. 7 of those are multi-hop retrieval tasks, where the gate answered after the first
  lookup. Every probe task is a first-step decision (the pack has 9 single-hop retrieval tasks and no
  later steps), so L's profile never saw "one lookup done, another needed".
- Removing tools made gpt-oss write longer answers (completion tokens 36k → 58k over the suite; output
  costs 7× input), and it called removed tools 20 times, each costing a second request. Fewer calls did
  not mean fewer dollars.
- Where L is genuinely capable (calculator, code, matrix and prime tasks), calls fell from 18 to 6 with no
  lost success.

### Per-agent calibration matters

Offline, on held-out When2Tool tasks labeled by measuring each agent (false skip = tools withheld from a
point that needed them):

| | Shared, no profile | Own profile (shipped) | Other agent's profile |
|---|---|---|---|
| S false skips | 65/70 | **4/70** | 42/70 |
| L false skips | 13/17 | **1/17** | 3/17 |

End to end, giving S the profile fitted for L cost 32 of 150 successes (98 vs 130). The same request can be
ANSWER for one agent and ACT for another; one global threshold cannot serve both.

The shipped profiles are conservative: on the same held-out sets they skipped 86 of L's 129 and 23 of S's
66 unnecessary calls, and on hand-written conversations (E3) only 19/62 and 14/60. A comparison model with
the agent's ID in its training input skipped far more (125/129 for L) with several times the false skips
(6/17; S 11/70).

### Agent-agnostic routing (When2Call)

On 900 held-out When2Call MCQ items the shared backbone reaches macro-F1 0.642, against 0.519 for untuned
FunctionGemma, 0.281 for the prompted Qwen router and 0.197 for rules. Its validation score was 0.925: the
official test distribution is harder than our training-derived validation split. Official-code When2Call
and BFCL relevance results: see [REPORT_final.md](eval/results/REPORT_final.md).

### Cost of running Whetherman

| | GPU (GTX 1660 Ti) p50 / p95 | CPU p50 / p95 |
|---|---|---|
| Short request, 2 tools (128 tokens) | 55 / 105 ms | 188 / 204 ms |
| 8 tools (830 tokens) | 122 / 127 ms | 1075 / 1203 ms |
| Long history (1504 tokens) | 249 / 410 ms | 2152 / 3434 ms |

Calibration: L $0.24 (83 min of agent time), U $0.35 (154 min), S 19 min locally, about 120 usable probe
labels each. At the measured savings, U's calibration pays back after roughly 24,000 episodes; L's does not
pay back. Total hosted spend for the whole project: see `eval/results/spend.json`.

## Limitations

- **Two agents, one day each.** Labels are measured evidence about one agent on one serving stack on one
  day (temperature 0), not ground truth. A different quantization, template or provider is a different agent;
  recalibrate.
- **Small denominators.** The probe packs give each agent about 120 usable labels; held-out sets have tens
  of ACT items per agent. Confidence intervals are wide and are reported.
- **Probe-pack coverage.** Calibration only knows the task shapes in the probe pack, and every probe task is
  a first-step decision. Later steps of multi-hop retrieval are where the large agent's false skips were.
- **CPU latency.** On CPU the router adds about 1 s per model call with 8 tools; use a GPU or accept it.
- **Generalization gap.** When2Call test macro-F1 (0.642) is far below validation (0.925).
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
