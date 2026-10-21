import json

from whetherman.tools.registry import ToolEnv, load_all

load_all()


def run(names, name, args, context=None):
    env = ToolEnv(names, context=context or {})
    return json.loads(env.execute(name, json.dumps(args))), env


def test_bigint_calculator():
    out, env = run(["evaluate_expression"], "evaluate_expression",
                   {"expression": "(40880268647*882767139926)-17904672"})
    assert out["result"] == 36087757832918701795450
    assert env.calls[0].ok


def test_calculator_rejects_code():
    out, env = run(["calculator"], "calculator", {"expression": "__import__('os').system('x')"})
    assert "error" in out and not env.calls[0].ok


def test_unknown_tool_and_bad_json():
    env = ToolEnv(["calculator"])
    assert "error" in json.loads(env.execute("rm_rf", "{}"))
    assert "error" in json.loads(env.execute("calculator", "{not json"))
    assert len(env.calls) == 2


def test_math_tools():
    assert run(["combination"], "combination", {"n": 83, "k": 12})[0]["result"] == 96851050941060
    assert run(["matrix_determinant"], "matrix_determinant",
               {"matrix": [[-12, 11, -12, -3], [-12, 6, 26, 3], [7, 22, 15, -19],
                           [24, 24, -15, -3]]})[0]["result"] == 372045
    assert run(["is_prime"], "is_prime", {"n": 799582})[0]["result"] is False
    assert run(["factorize"], "factorize", {"n": 360})[0]["result"] == "2^3 * 3^2 * 5"
    assert run(["nth_prime"], "nth_prime", {"n": 100})[0]["result"] == 541
    assert run(["day_of_week"], "day_of_week", {"date": "2017-09-01"})[0]["day"] == "Friday"
    assert run(["encode"], "encode", {"message": "HAWK", "encoding_type": "morse"})[0][
        "encoded"] == ".... .- .-- -.-"


def test_schedule_counts_gaps():
    out, _ = run(["find_free_slots"], "find_free_slots",
                 {"busy": [["09:09", "10:39"], ["10:41", "11:11"]], "day_start": "09:00",
                  "day_end": "17:00", "min_duration": 60})
    assert out["count"] == 1


def test_run_code():
    out, _ = run(["run_code"], "run_code", {"code": "print(sum(range(10)))"})
    assert out["stdout"].strip() == "45"


def test_dev_tools_and_side_effects():
    names = ["read_file", "search_repo", "run_tests", "send_email", "web_search", "get_weather"]
    out, _ = run(names, "read_file", {"path": "shopcart/pricing.py"})
    assert "TAX_RATE = 0.0725" in out["content"]
    out, _ = run(names, "read_file", {"path": "../../../etc/passwd"})
    assert "error" in out
    out, _ = run(names, "search_repo", {"query": "MAX_ITEMS"})
    assert any("config.py" in m for m in out["matches"])
    out, _ = run(names, "run_tests", {})
    assert out["exit_code"] == 1 and "1 failed, 5 passed" in out["output"]
    out, env = run(names, "send_email", {"to": "a@b.c", "subject": "hi", "body": "x"})
    assert env.state["emails"][0]["to"] == "a@b.c"
    assert run(names, "web_search", {"query": "tidewater latest version"})[0]["results"]
    assert run(names, "web_search", {"query": "zzz qqq"})[0]["results"] == []
    assert run(names, "get_weather", {"city": "Oslo"})[0]["temp_c"] == 4.5
