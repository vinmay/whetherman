from itertools import cycle

from whetherman.episode import Episode
from whetherman.labeling import label_necessity, load_config, wilson
from whetherman.tasks import Task

TASK = Task(id="t", messages=[{"role": "user", "content": "q"}], tools=["calculator"],
            oracle={"type": "answer", "gold": "1"})


def fake(no_seq, tool_seq):
    no_it, tool_it = cycle(no_seq), cycle(tool_seq)
    calls = {"n": 0}

    def run_fn(agent, task, with_tools, **kw):
        calls["n"] += 1
        ok = next(tool_it) if with_tools else next(no_it)
        return Episode(task_id=task.id, with_tools=with_tools, success=ok, final_text="")

    return run_fn, calls


def lab(no_seq, tool_seq):
    run_fn, calls = fake(no_seq, tool_seq)
    return label_necessity(TASK, agent=None, config=load_config(), run_fn=run_fn), calls["n"]


def test_answer_on_screen():
    lbl, n = lab([1, 1, 1], [1, 1, 1])
    assert lbl.label == "ANSWER" and lbl.stage == "screen" and n == 6


def test_act_on_screen():
    lbl, _ = lab([0, 0, 0], [1, 0, 1])
    assert lbl.label == "ACT" and lbl.tool_success == 2


def test_mixed_extends_then_answer():
    lbl, n = lab([1, 1, 0, 1, 1], [1])  # 4/5 without tools -> not >= 5 -> ambiguous
    assert lbl.stage == "extended" and n == 10 and lbl.label == "ambiguous"


def test_mixed_extends_then_act():
    lbl, _ = lab([1, 0, 0, 0, 0], [1, 1, 0, 1, 1])  # 1/5 without, 4/5 with
    assert lbl.label == "ACT" and lbl.stage == "extended"


def test_neither_condition_works_is_ambiguous():
    lbl, _ = lab([0], [0])
    assert lbl.label == "ambiguous"


def test_wilson():
    lo, hi = wilson(0, 8)
    assert lo == 0 and 0.3 < hi < 0.33
    assert wilson(0, 0) == (0.0, 1.0)
