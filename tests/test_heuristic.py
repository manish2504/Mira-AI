from src.heuristic import heuristic_extract


def test_heuristic_favorite_and_switch():
    first = heuristic_extract("My favorite drink is black coffee.")
    assert first["memories"][0]["key"] == "favorite drink"
    assert first["memories"][0]["value"].lower() == "black coffee"
    second = heuristic_extract("I've switched to green tea recently.")
    assert second["memories"][0]["value"].lower() == "green tea"


def test_heuristic_employer_update():
    a = heuristic_extract("I work at Microsoft and I really like backend engineering.")
    assert a["memories"][0]["value"] == "Microsoft"
    b = heuristic_extract("I left Microsoft and joined Google.")
    assert b["memories"][0]["value"] == "Google"


def test_heuristic_ignores_smalltalk():
    assert heuristic_extract("How was your morning?")["memories"] == []
