# Data

Every file below is produced by a committed script with a fixed seed. Hashes are in
`data/fixtures/manifest.json`, `data/fixtures/ab/manifest.json` and `data/fixtures/e3/e3.sha256`.

## Sources and licenses

| Source | License | Used for | Redistributed here? |
|---|---|---|---|
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | E1 train/val (from `train_pref` + `train_sft`), E1 test (from the official MCQ test) | Yes: converted subsets, with attribution |
| [cesun/When2Tool](https://huggingface.co/datasets/cesun/When2Tool) ([paper](https://arxiv.org/abs/2605.09252)) | MIT | E2 task pools, probe tasks, part of the A/B suites | Yes: converted task subsets |
| Hand-written (this repo) | Apache-2.0 | E3, multi-step training templates, probe-pack developer tasks, A/B developer tasks, fixture repo | Yes |

Raw downloads live in `data/raw/` (gitignored); `uv run python data/prepare.py --download` refetches them.

When2Tool ships tool *names* but no schemas or license for its environment code, so Whetherman
reimplements the tools with its own schemas (`whetherman/tools/w2t.py`). Tasks that rely on the
benchmark's private "custom" hash/cipher algorithms (40 of 900 single-hop train items, 100 of 2,250 test
items) and the `ListManipulationEnv` environment (60 train, 150 test items) are excluded.

## Conversion (`data/prepare.py`)

- **E1 labels.** When2Call MCQ test items carry an explicit `correct_answer` (`tool_call` → ACT,
  `request_for_info` → ASK, `cannot_answer` → DECLINE). The training files carry no explicit label:
  a response containing `<TOOLCALL>` is ACT; otherwise a refusal phrase in its first 160 characters
  (apolog/sorry/unable/can't/cannot/...) means DECLINE, else ASK. This ASK/DECLINE split is a
  heuristic and is documented as such. Items with no tools are dropped (a decision point requires tools).
- **E1 sizes.** Train 4,500 and val 600 (balanced, disjoint, sampled from the 24,000 training rows);
  test 900 (balanced, stratified sample of the 3,652-item official MCQ test). The full MCQ file is
  kept in `data/raw/` for the official evaluation.
- **Tool schemas** (BFCL style → OpenAI): `dict`→`object`, `str`→`string`, `int`→`integer`,
  `float`→`number`, `bool`→`boolean`, `list`/`tuple`→`array`; ", optional" suffixes stripped.
- **When2Tool pools.** Stratified by environment × difficulty: `e2_train` 200 and `probe` 120 from the
  train split (disjoint), `e2_test` 160 from the test split; multi-hop `multihop_train` 54 / `multihop_test` 36.

## Splits and discipline

| Set | File(s) | Use | May influence implementation? |
|---|---|---|---|
| Train | `e1/train`, `train/multistep`, E2-train items where agents agree | backbone | yes |
| Validation | `e1/val`, held-out multi-step families, E2-train agreement val, E2-train disagreement items | backbone selection, gates | yes |
| Probe (per agent) | `probe_pack/` (120 When2Tool + 30 developer tasks) | adaptation 60% / threshold 40% | yes |
| Development A/B | `ab/ab_dev.jsonl` (42 tasks, 20 multi-step) | debugging | yes |
| E1/E2 held-out test | `e1/test`, `w2t/e2_test`, `w2t/multihop_test` | final offline benchmark | **no** |
| E3 | `e3/e3.jsonl` (123 decision points, 84 conversations, 27 multi-step, 23 unseen-schema points) | final generalization | **no** |
| Final A/B | `ab/ab_final.jsonl` (80 tasks, 40 multi-step) | final end-to-end | **no** |

Labeling held-out sets with `label_necessity` runs the *agent* on them to produce labels; no router is
evaluated and nothing is tuned on those labels before the freeze.

### Held-out exposure record

- 2026-10-21: three E3 decision points (`e3-repo-currency-s0`, `e3-un-cancel-s0`, `e3-ask-report-s0`)
  were run through the prompted Qwen3.5-4B router and the rules router to confirm that Ollama returns
  log-probabilities. Outputs were seen; no code, prompt, threshold or data was changed in response.

## Tool-necessity labels

`whetherman/labeling.py::label_necessity` with `configs/labeling.yaml` (shared by E2 and `calibrate`):
3 trials without tools + 3 with tools; ANSWER if 3/3 succeed without tools; ACT if 0/3 without and ≥2/3
with; otherwise 2 more trials each (5 total): ANSWER if 5/5 without, ACT if ≤1/5 without and ≥4/5 with,
else **ambiguous** (excluded from training/adaptation, reported separately). Each label stores trial counts,
success rates and Wilson 95% intervals. Decoding matches deployment (temperature 0, a different seed per
trial). **These labels are measured evidence about one agent on one day, not ground truth.**

Side-effect tasks only succeed if the simulated state changed correctly (an agent that says "email sent"
without calling the tool fails).

## Deduplication

See `eval/results/dedup.json` (normalized-text hash plus embedding cosine > 0.95 across all splits).
