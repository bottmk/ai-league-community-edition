# CodeRunner Tool - AI League
# ===========================
# Executes Python the agent writes and returns what it printed.
#
# Exists for the challenge tiles that pose a calculation a language model
# cannot do in its head (c2 Blue Brain, and the arithmetic half of the c6
# boss). The canonical example is the last ten digits of the 3000th Fibonacci
# number: a few thousand big-integer additions, instant here, hopeless as a
# next-token guess. The agent writes the program, this runs it, and the printed
# output is the answer.
#
# Rules this tool is shaped by, from the official AI League rules page:
#   - No Bedrock, no OpenAI, no model API of any kind may be called from a
#     Lambda tool. Nothing here calls a model; it runs the caller's Python.
#   - Hardcoding answers is a disqualification, so there is no lookup table of
#     questions or answers here: whatever comes back was computed just now.
#   - Calling an external site the challenge did not name is a disqualification,
#     so this tool never opens a socket of its own and has no URL in it.
#   - No extra libraries may be installed. Standard library only.
#
# Running caller-supplied code is the entire point of the tool, so exec() is
# deliberate, not an oversight. What is guarded is the clock and the output
# size, because the Lambda has 30 seconds and the run's token bonus is
# 1000 - (total tokens / challenges visited): a runaway loop would burn the
# invocation, and an unbounded print would burn the bonus.
#
# Deployed as the AgentCoreGatewayTool-<name> Lambda behind the AgentCore
# Gateway. The gateway hands the tool arguments in flat, at the top level of
# the event ({"code": "print(2 + 2)"}), and expects
# {"statusCode": 200, "body": json.dumps(result)} back. The 'body' unwrapping
# below is a fallback for an API Gateway style invocation.
#
# In the AI League environment this file is named lambda_function.py, so the
# handler setting reads lambda_function.lambda_handler.

import json
import signal
import sys
import threading
import time

# Wall-clock budget for the caller's code, comfortably inside the Lambda's 30
# seconds so that a timeout still returns a JSON body the agent can react to.
TIME_LIMIT_SECONDS = 12.0

# Printed output is trimmed to this, because every returned character is paid
# for out of the token bonus.
MAX_OUTPUT_CHARS = 20000

CODE_FILENAME = '<agent_code>'


class _TimeLimitExceeded(BaseException):
    """Raised inside the caller's code when its time is up.

    Deliberately a BaseException: a bare `except Exception` in generated code
    must not be able to swallow the deadline and keep spinning.
    """


def lambda_handler(event, context):
    """
    AWS Lambda function that executes Python code and returns its printed output.
    Handles both API Gateway format and direct AgentCore Gateway format.

    ---
    Tool: run_python
    Description: Runs Python code and returns everything it printed. Use it for any calculation, string manipulation or big-number arithmetic instead of working the answer out in your head. The code must print its final answer.
    Parameters:
        code  (required) - the Python program to run. It must print() the answer. The standard library is importable. Do not fetch pages from here; that is the web fetch tool's job
    ---

    ## Return
        result    - everything the code printed, trimmed
        status    - ok, error, or timeout
        error     - present only when the code raised or ran out of time, as
                    "ExceptionType: message", with no traceback
        truncated - true when the printed output was longer than the cap

    Code that raises still returns 200 with an 'error', so the agent can read
    the message, fix the program and try again. Only a missing 'code'
    parameter is a non-200.

    ## Example
        code: "a, b = 0, 1\\nfor _ in range(3000): a, b = b, a + b\\nprint(a % 10**10)"
        returns the last ten digits of the 3000th Fibonacci number.
    """

    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        code = body.get('code')
        if code is None:
            code = body.get('source') or body.get('python')
        if isinstance(code, (list, tuple)):
            code = '\n'.join(str(line) for line in code)
        code = '' if code is None else str(code)
        code = _strip_code_fence(code)

        if not code.strip():
            return _err(400, 'Missing required parameter: code')

        print(f"DEBUG: running {len(code)} characters of code")
        result = run_code(code)
        print(f"RESULT: status={result['status']} chars={len(result['result'])}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:  # noqa: BLE001 - the gateway needs a JSON body, never a stack
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


def _strip_code_fence(code):
    """Drop a markdown fence, which a model wraps code in more often than not."""
    text = code.strip()
    if not text.startswith('```'):
        return code
    lines = text.split('\n')
    lines = lines[1:]
    while lines and lines[-1].strip().startswith('```'):
        lines.pop()
    return '\n'.join(lines)


class _BoundedOutput:
    """Stands in for sys.stdout, keeping at most MAX_OUTPUT_CHARS.

    Past the cap the writes are counted and dropped, so `while True: print(x)`
    costs time but not memory.
    """

    def __init__(self, limit):
        self._limit = limit
        self._parts = []
        self._length = 0
        self.truncated = False

    def write(self, text):
        text = text if isinstance(text, str) else str(text)
        remaining = self._limit - self._length
        if remaining <= 0:
            self.truncated = True
            return len(text)
        if len(text) > remaining:
            self._parts.append(text[:remaining])
            self._length = self._limit
            self.truncated = True
        else:
            self._parts.append(text)
            self._length += len(text)
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        return None

    def isatty(self):
        return False

    def getvalue(self):
        return ''.join(self._parts)


def _describe(error):
    """The exception as one line, with no traceback noise."""
    message = str(error).strip()
    name = type(error).__name__
    return f'{name}: {message}' if message else name


def run_code(code, time_limit=None):
    """Execute code, returning the result dict the handler serialises.

    Never raises: a failure in the caller's code is a result, not an error,
    because the agent's next move is to read the message and retry.
    """
    limit = TIME_LIMIT_SECONDS if time_limit is None else float(time_limit)
    captured = _BoundedOutput(MAX_OUTPUT_CHARS)

    # One namespace for globals and locals. Two separate dicts would put a
    # def at top level into the locals while name lookup inside its body went
    # to the globals, so any recursive or mutually referring function the agent
    # writes would raise NameError. That is exactly the shape of code a
    # Fibonacci question invites, so it has to work.
    namespace = {'__builtins__': __builtins__, '__name__': '__main__'}

    started = time.monotonic()
    failure = None
    timed_out = False

    original_stdout = sys.stdout
    sys.stdout = captured
    try:
        compiled = compile(code, CODE_FILENAME, 'exec')
    except SyntaxError as syntax_error:
        sys.stdout = original_stdout
        return {
            'result': '',
            'status': 'error',
            'error': _describe(syntax_error),
            'truncated': False,
        }
    except (ValueError, MemoryError) as compile_error:
        sys.stdout = original_stdout
        return {
            'result': '',
            'status': 'error',
            'error': _describe(compile_error),
            'truncated': False,
        }

    try:
        with _deadline(limit):
            exec(compiled, namespace)  # noqa: S102 - running the agent's code is the point
    except _TimeLimitExceeded:
        timed_out = True
    except SystemExit:
        pass  # sys.exit() in generated code is a normal finish
    except KeyboardInterrupt:
        timed_out = True
    except BaseException as error:  # noqa: BLE001 - every failure is a result
        failure = _describe(error)
    finally:
        sys.stdout = original_stdout

    elapsed = time.monotonic() - started
    if not timed_out and failure is None and elapsed > limit:
        # The deadline fired somewhere the exception could not be delivered,
        # or the guard was unavailable. Report it rather than pretend.
        timed_out = True

    output = captured.getvalue().strip()
    result = {
        'result': output,
        'status': 'ok',
        'truncated': captured.truncated,
    }
    if timed_out:
        result['status'] = 'timeout'
        result['error'] = (
            f'Execution stopped after {limit:g} seconds. Any output above is partial; '
            f'rewrite the code to finish faster.'
        )
    elif failure is not None:
        result['status'] = 'error'
        result['error'] = failure
    return result


# ---------------------------------------------------------------------------
# Wall-clock guard
#
# SIGALRM is the cheap way to interrupt a runaway loop: it costs nothing while
# the code runs and raises inside it when the time is up. It only works on the
# main thread, which is where Lambda calls the handler, so a trace-based
# fallback covers the case where the tool is driven from a worker thread.
# ---------------------------------------------------------------------------


class _deadline:
    """Context manager that raises _TimeLimitExceeded in the body after `limit`."""

    def __init__(self, limit):
        self.limit = max(0.05, float(limit))
        self._previous_handler = None
        self._previous_tracer = None
        self._armed = False
        self._traced = False

    def __enter__(self):
        if hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread():
            try:
                self._previous_handler = signal.signal(signal.SIGALRM, self._fire)
                signal.setitimer(signal.ITIMER_REAL, self.limit)
                self._armed = True
                return self
            except (ValueError, OSError):
                self._armed = False
        self._start_trace()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._previous_handler is not None:
                signal.signal(signal.SIGALRM, self._previous_handler)
        if self._traced:
            sys.settrace(self._previous_tracer)
        return False

    def _fire(self, signum, frame):
        raise _TimeLimitExceeded()

    def _start_trace(self):
        """Fallback guard: check the clock on every line the caller's code runs."""
        expires_at = time.monotonic() + self.limit

        def tracer(frame, event, arg):
            if time.monotonic() > expires_at:
                raise _TimeLimitExceeded()
            return tracer

        self._previous_tracer = sys.gettrace()
        sys.settrace(tracer)
        self._traced = True
