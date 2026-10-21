from whetherman.tasks import answer_matches, check


def test_numbers():
    assert answer_matches("The answer is **40**.", "40")
    assert answer_matches("7.5", "7.50")
    assert answer_matches("36,087,757,832,918,701,795,450", "36087757832918701795450")
    assert not answer_matches("41", "40")
    assert answer_matches("0.98951", "0.9895")
    assert not answer_matches("0.9899", "0.9895")


def test_structs_and_text():
    assert answer_matches("['!', '@']", "['!', '@']")
    assert answer_matches('Result: ["!", "@"]', "['!', '@']")
    assert answer_matches("[8.0, 21, 7]", "[8, 21, 7]")
    assert answer_matches("It was Johannes Gutenberg.", "Johannes Gutenberg")
    assert answer_matches("False", "False")
    assert not answer_matches("True or False? False", "False")
    assert answer_matches("<think>hmm</think>Friday", "Friday")
    assert not answer_matches("", "x")


def test_state_oracle_requires_change():
    o = {"type": "state", "emails": [{"to": "ops@shopcart.dev", "subject__contains": "release"}]}
    assert not check(o, "Email sent!", {"emails": [], "issues": []})
    st = {"emails": [{"to": "ops@shopcart.dev", "subject": "Release notes", "body": ""}],
          "issues": []}
    assert check(o, "", st)


def test_question_and_decline():
    assert check({"type": "question"}, "Which city do you mean?", {})
    assert not check({"type": "question"}, "It is sunny.", {})
    assert check({"type": "decline"}, "I can't book flights with the tools I have.", {})
