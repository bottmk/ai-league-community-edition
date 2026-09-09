"""Tests for the CodeRunner tool.

CodeRunner answers the challenge tiles that pose arithmetic a language model
cannot do in its head (c2 Blue Brain, and the calculation half of the c6 boss).
The in-game example is the last ten digits of the 3000th Fibonacci number, so
that case is computed here independently and compared.

Two things are being protected. The first is the deployment contract: the
AgentCore Gateway hands tool arguments in flat at the top level of the event
and expects {'statusCode': 200, 'body': json.dumps(result)} back, inside a 30
second Lambda timeout. The second is the official rules: a Lambda tool may not
call Bedrock or any other model API, may not contact a site the challenge did
not name, and may not hardcode answers. Each of those is scanned from the
module source, because each one disqualifies the entrant.
"""

import ast
import importlib.util
import io
import json
import os
import sys
import time
import tokenize

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TOOL_PATH = os.path.join(_REPO_ROOT, "agent-config", "tools", "CodeRunner", "index.py")


def _code_only(source):
    """The module's executable text, with comments and docstrings removed.

    The comments name the rules the tool obeys, Bedrock included; only what
    actually runs is scanned below.
    """
    kept = []
    previous_type = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous_type in (
            tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL
        ):
            continue  # a docstring
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.COMMENT):
            previous_type = token.type
    return "\n".join(kept)


def _load():
    spec = importlib.util.spec_from_file_location("code_runner_tool", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def code_runner():
    if not os.path.exists(_TOOL_PATH):
        pytest.skip(f"tool not present: {_TOOL_PATH}")
    return _load()


def _run(code_runner, code, **arguments):
    payload = {"code": code}
    payload.update(arguments)
    response = code_runner.lambda_handler(payload, None)
    return response["statusCode"], json.loads(response["body"])


def _fibonacci(n):
    """F(n) by fast doubling — a different algorithm from the one under test."""
    def pair(k):
        if k == 0:
            return 0, 1
        a, b = pair(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (d, c + d) if k & 1 else (c, d)

    return pair(n)[0]


class TestGatewayContract:
    def test_flat_event_returns_a_json_body(self, code_runner):
        response = code_runner.lambda_handler({"code": "print(6 * 7)"}, None)
        assert response["statusCode"] == 200
        assert isinstance(response["body"], str)
        assert json.loads(response["body"])["result"] == "42"

    def test_api_gateway_shaped_event_is_unwrapped(self, code_runner):
        for body in (json.dumps({"code": "print(6 * 7)"}), {"code": "print(6 * 7)"}):
            response = code_runner.lambda_handler({"body": body}, None)
            assert response["statusCode"] == 200
            assert json.loads(response["body"])["result"] == "42"

    def test_missing_code_is_a_non_200_with_an_error(self, code_runner):
        for event in ({}, {"code": ""}, {"code": "   \n  "}):
            response = code_runner.lambda_handler(event, None)
            assert response["statusCode"] == 400
            assert "code" in json.loads(response["body"])["error"].lower()

    def test_a_markdown_fence_is_stripped(self, code_runner):
        """A model wraps code in a fence more often than not."""
        status, result = _run(code_runner, "```python\nprint(6 * 7)\n```")
        assert status == 200
        assert result["result"] == "42"


class TestExecution:
    def test_printed_output_is_returned_trimmed(self, code_runner):
        status, result = _run(code_runner, "print('  hello  ')\n")
        assert status == 200
        assert result["result"] == "hello"
        assert result["status"] == "ok"

    def test_several_prints_are_kept_in_order(self, code_runner):
        status, result = _run(code_runner, "for i in range(3):\n    print(i)")
        assert status == 200
        assert result["result"].split() == ["0", "1", "2"]

    def test_the_standard_library_is_importable(self, code_runner):
        status, result = _run(code_runner, "import math\nprint(math.isqrt(1024))")
        assert status == 200
        assert result["result"] == "32"

    def test_a_function_can_call_itself(self, code_runner):
        """A single namespace for globals and locals, or this raises NameError.

        Recursion is exactly the shape of code a Fibonacci question invites, so
        it has to work.
        """
        status, result = _run(
            code_runner,
            "def f(n):\n    return 1 if n < 2 else f(n - 1) + f(n - 2)\nprint(f(15))",
        )
        assert status == 200, result
        assert result["result"] == "987"


class TestTheBlueBrainWorkload:
    """c2's worked example: F(3000), last ten digits, F(0)=0, F(1)=1."""

    def test_the_three_thousandth_fibonacci_number_is_computed(self, code_runner):
        started = time.monotonic()
        status, result = _run(
            code_runner,
            "a, b = 0, 1\n"
            "for _ in range(3000):\n"
            "    a, b = b, a + b\n"
            "print(str(a)[-10:])",
        )
        elapsed = time.monotonic() - started

        assert status == 200, result
        assert result["status"] == "ok"
        assert result["result"] == str(_fibonacci(3000))[-10:]
        assert elapsed < 5  # the Lambda budget is 30 seconds

    def test_arbitrary_precision_arithmetic_survives_the_json_round_trip(self, code_runner):
        status, result = _run(code_runner, "print(2 ** 4096 % 1000003)")
        assert status == 200
        assert result["result"] == str(pow(2, 4096, 1000003))


class TestFailureIsAResultNotACrash:
    def test_a_raising_snippet_returns_an_error_not_a_500(self, code_runner):
        status, result = _run(code_runner, "raise ValueError('boom')")
        assert status == 200, result
        assert result["status"] == "error"
        assert result["error"] == "ValueError: boom"

    def test_the_error_carries_no_traceback_noise(self, code_runner):
        status, result = _run(code_runner, "1 / 0")
        assert status == 200
        assert "Traceback" not in result["error"]
        assert "File \"" not in result["error"]
        assert result["error"].startswith("ZeroDivisionError")

    def test_a_syntax_error_is_reported_with_its_line(self, code_runner):
        status, result = _run(code_runner, "print('unclosed'")
        assert status == 200
        assert result["status"] == "error"
        assert "SyntaxError" in result["error"]

    def test_a_name_error_names_the_missing_name(self, code_runner):
        """The message is what the agent reads before rewriting the code."""
        status, result = _run(code_runner, "print(nonexistent_thing)")
        assert status == 200
        assert "nonexistent_thing" in result["error"]

    def test_output_before_a_failure_is_kept(self, code_runner):
        status, result = _run(code_runner, "print('partial')\nraise RuntimeError('later')")
        assert status == 200
        assert result["result"] == "partial"
        assert result["status"] == "error"

    def test_sys_exit_is_a_normal_finish(self, code_runner):
        status, result = _run(code_runner, "import sys\nprint('done')\nsys.exit(0)")
        assert status == 200
        assert result["status"] == "ok"
        assert result["result"] == "done"


class TestTheWallClockGuard:
    """A runaway loop must cost a few seconds, not the whole invocation."""

    def test_a_runaway_loop_is_cut_off(self, code_runner):
        started = time.monotonic()
        result = code_runner.run_code("while True:\n    pass", time_limit=1.0)
        elapsed = time.monotonic() - started
        assert result["status"] == "timeout"
        assert elapsed < 5
        assert "seconds" in result["error"]

    def test_the_deadline_cannot_be_swallowed_by_the_running_code(self, code_runner):
        """The timeout is a BaseException, so `except Exception` cannot eat it."""
        started = time.monotonic()
        result = code_runner.run_code(
            "while True:\n    try:\n        pass\n    except Exception:\n        pass",
            time_limit=1.0,
        )
        assert result["status"] == "timeout"
        assert time.monotonic() - started < 5

    def test_the_handler_honours_the_module_time_limit(self, code_runner, monkeypatch):
        monkeypatch.setattr(code_runner, "TIME_LIMIT_SECONDS", 1.0)
        started = time.monotonic()
        status, result = _run(code_runner, "while True:\n    pass")
        assert status == 200
        assert result["status"] == "timeout"
        assert time.monotonic() - started < 5

    def test_the_default_limit_leaves_room_inside_the_lambda_timeout(self, code_runner):
        assert 0 < code_runner.TIME_LIMIT_SECONDS < 30

    def test_the_guard_is_torn_down_after_a_normal_run(self, code_runner):
        """A leftover timer or trace function would fire during the next run."""
        code_runner.run_code("print('quick')", time_limit=1.0)
        assert sys.gettrace() is None
        time.sleep(0.05)
        status, result = _run(code_runner, "print('still fine')")
        assert status == 200
        assert result["result"] == "still fine"

    def test_stdout_is_restored_afterwards(self, code_runner):
        original = sys.stdout
        code_runner.run_code("print('x')\nraise RuntimeError('boom')", time_limit=1.0)
        assert sys.stdout is original


class TestTokenEconomy:
    """The run bonus is 1000 - tokens/challenges, so the return stays bounded."""

    def test_a_flood_of_output_is_capped(self, code_runner):
        started = time.monotonic()
        result = code_runner.run_code("while True:\n    print('x' * 100)", time_limit=1.0)
        assert result["truncated"] is True
        assert len(result["result"]) <= code_runner.MAX_OUTPUT_CHARS
        assert time.monotonic() - started < 5

    def test_a_short_answer_is_not_marked_truncated(self, code_runner):
        status, result = _run(code_runner, "print('42')")
        assert result["truncated"] is False
        assert status == 200


class TestDisqualificationRules:
    """Scanned from the source, because each of these ends the entry."""

    @pytest.fixture
    def source(self):
        with open(_TOOL_PATH, encoding="utf-8") as handle:
            return handle.read()

    def test_no_url_is_hardcoded(self, source):
        """The tool never contacts a site of its own."""
        assert "://" not in source

    def test_no_model_or_aws_api_is_called(self, source):
        lowered = _code_only(source).lower()
        for forbidden in ("boto3", "bedrock", "invoke_model", "openai", "anthropic",
                          "converse(", "sagemaker", "urlopen", "socket"):
            assert forbidden not in lowered, forbidden

    def test_only_the_standard_library_is_imported(self, source):
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
        for banned in ("requests", "numpy", "sympy", "httpx"):
            assert banned not in imported

    def test_no_answer_is_hardcoded(self, source):
        """The Blue Brain answer must be computed, never looked up."""
        assert str(_fibonacci(3000))[-10:] not in source
        lowered = _code_only(source).lower()
        for smell in ("answers = {", "answer_table", "known_answers", "fibonacci"):
            assert smell not in lowered, smell
