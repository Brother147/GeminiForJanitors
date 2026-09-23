from gfjproxy.commands import Command, parse_message, strip_message


def test_parse_message():
    assert parse_message("Hello, World!") == ([], "Hello, World!")
    assert parse_message("//banner") == ([Command("banner")], "")
    assert parse_message("//fixturns on") == ([Command("fixturns", "on")], "")
    assert parse_message("//roll 2d20 Attack") == ([Command("roll", "2d20")], "Attack")


def test_removed_commands_are_not_parsed():
    commands, content = parse_message("//prefill on hello //btrick on //think on //context //max_tokens 1000 //status")
    assert commands == []
    assert content == "//prefill on hello //btrick on //think on"


def test_strip_message():
    assert strip_message("  A\nB  ") == "A\nB"
