"""Tests for the WebFetch tool.

WebFetch answers the challenge tiles that quote a URL and ask for a fact
printed on the page (c4 web scraping, c23 web fetch, and the fetch half of the
c6 boss). Two things are being protected here.

The first is the deployment contract: the AgentCore Gateway hands tool
arguments in flat at the top level of the event and expects
{'statusCode': 200, 'body': json.dumps(result)} back.

The second is the official rules. Contacting a site the challenge did not name
is a disqualification, so no test here is allowed to touch the network:
urllib.request.urlopen and the module's own DNS lookup are both stubbed, and
the module source is scanned for hardcoded URLs, model API calls and
third-party imports, each of which would disqualify the entrant.
"""

import ast
import email.message
import io
import tokenize
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TOOL_PATH = os.path.join(_REPO_ROOT, "agent-config", "tools", "WebFetch", "index.py")

# A public address, so the destination check passes without a DNS lookup.
_PUBLIC_IP = "93.184.216.34"

PAGE = b"""<html><head><title>Ada &amp; Friends</title>
<style>.hidden { display: none }</style>
<script>var decoy = "her hobby is writing javascript";</script>
</head><body>
<h1>Profile</h1>
<p>Ada&#39;s hobby is <b>rock climbing</b>.</p>
<ul><li>Joined 1843</li></ul>
<p>""" + b"padding sentence. " * 40 + b"""</p>
</body></html>"""

# What a JavaScript-rendered page looks like on the wire: a mount point, a
# bundle, and no answer anywhere in the HTML.
EMPTY_PAGE = b'<html><body><div id="root"></div><script>window.render()</script></body></html>'


def _code_only(source):
    """The module's executable text, with comments and docstrings removed.

    The comments explain which rules the tool obeys and so mention Bedrock by
    name; only what actually runs is scanned below.
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
    spec = importlib.util.spec_from_file_location("web_fetch_tool", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    """The parts of an http.client.HTTPResponse the tool actually touches."""

    def __init__(self, data, content_type="text/html; charset=utf-8", url="https://site.test/p"):
        self._data = data
        self._url = url
        self.headers = email.message.Message()
        if content_type:
            self.headers["Content-Type"] = content_type

    def read(self, size=None):
        return self._data[:size] if size is not None else self._data

    def geturl(self):
        return self._url

    def close(self):
        return None


@pytest.fixture
def web_fetch(monkeypatch):
    """The tool, with the network unplugged: no DNS, no sockets."""
    if not os.path.exists(_TOOL_PATH):
        pytest.skip(f"tool not present: {_TOOL_PATH}")
    module = _load()

    def _no_dns(host):
        return [_PUBLIC_IP]

    def _no_network(request, timeout=None):
        raise AssertionError("the test made a real network call")

    monkeypatch.setattr(module, "_resolve_addresses", _no_dns)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    return module


def _serve(monkeypatch, body, content_type="text/html; charset=utf-8", url="https://site.test/p"):
    """Make the next fetch return this response instead of opening a socket."""
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(body, content_type, url)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return captured


def _call(web_fetch, **arguments):
    response = web_fetch.lambda_handler(dict(arguments), None)
    return response["statusCode"], json.loads(response["body"])


class TestGatewayContract:
    """Arguments arrive flat; the body comes back as a JSON string."""

    def test_flat_event_returns_a_json_body(self, web_fetch, monkeypatch):
        _serve(monkeypatch, PAGE)
        response = web_fetch.lambda_handler({"url": "https://site.test/p"}, None)
        assert response["statusCode"] == 200
        assert isinstance(response["body"], str)
        assert json.loads(response["body"])["status"] == "ok"

    def test_api_gateway_shaped_event_is_unwrapped(self, web_fetch, monkeypatch):
        _serve(monkeypatch, PAGE)
        for body in (json.dumps({"url": "https://site.test/p"}), {"url": "https://site.test/p"}):
            response = web_fetch.lambda_handler({"body": body}, None)
            assert response["statusCode"] == 200
            assert "rock climbing" in json.loads(response["body"])["text"]

    def test_a_missing_url_is_a_non_200_with_an_error(self, web_fetch):
        status, result = _call(web_fetch)
        assert status == 400
        assert "url" in result["error"].lower()

    def test_string_arguments_are_tolerated(self, web_fetch, monkeypatch):
        """The sub-agent fills these in, so numbers arrive as strings."""
        _serve(monkeypatch, PAGE)
        status, result = _call(web_fetch, url=" <https://site.test/p> ", max_chars="250")
        assert status == 200, result
        assert result["chars"] <= 250

    def test_the_fetch_timeout_stays_inside_the_lambda_budget(self, web_fetch, monkeypatch):
        captured = _serve(monkeypatch, PAGE)
        _call(web_fetch, url="https://site.test/p")
        assert 0 < captured["timeout"] < 30
        assert captured["request"].get_header("User-agent")


class TestDestinationRules:
    """Only the caller's public http(s) URL is ever contacted."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://files.test/a",
            "gopher://old.test/1",
            "data:text/html,<p>hi</p>",
            "javascript:alert(1)",
            "site.test/no-scheme",
        ],
    )
    def test_non_http_schemes_are_refused(self, web_fetch, url):
        status, result = _call(web_fetch, url=url)
        assert status == 400
        assert "http" in result["error"].lower()

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/x",
            "http://localhost:8080/x",
            "http://something.localhost/x",
            "https://10.0.0.5/x",
            "https://192.168.1.1/x",
            "http://172.16.0.9/x",
            "http://[::1]/x",
            "http://169.254.169.254/latest/meta-data/",  # the instance metadata service
            "http://0.0.0.0/x",
        ],
    )
    def test_private_and_loopback_addresses_are_refused(self, web_fetch, url):
        status, result = _call(web_fetch, url=url)
        assert status == 400
        assert "refus" in result["error"].lower()

    def test_a_public_name_resolving_to_a_private_address_is_refused(self, web_fetch, monkeypatch):
        monkeypatch.setattr(web_fetch, "_resolve_addresses", lambda host: ["10.1.2.3"])
        status, result = _call(web_fetch, url="https://looks-public.test/x")
        assert status == 400
        assert "10.1.2.3" in result["error"]

    def test_same_site_redirects_are_allowed_and_off_site_ones_are_not(self, web_fetch):
        same_site = [
            ("https://site.test/a", "https://www.site.test/b"),
            ("https://www.site.test/a", "https://site.test/b"),
            ("http://site.test/a", "https://site.test/a"),
            ("https://site.test/a", "https://site.test/b"),
        ]
        for current, following in same_site:
            assert web_fetch._same_site(current, following), (current, following)
        assert not web_fetch._same_site("https://site.test/a", "https://elsewhere.test/a")
        assert not web_fetch._same_site("https://site.test/a", "https://site.test.evil.test/a")

    def test_the_redirect_handler_refuses_to_leave_the_site(self, web_fetch):
        handler = web_fetch._SameSiteRedirectHandler()
        request = urllib.request.Request("https://site.test/a")
        headers = email.message.Message()
        with pytest.raises(web_fetch.FetchRefused) as refusal:
            handler.redirect_request(request, None, 302, "Found", headers,
                                     "https://elsewhere.test/b")
        assert "off-site" in str(refusal.value)

        followed = handler.redirect_request(request, None, 302, "Found", headers,
                                            "https://www.site.test/b")
        assert followed.full_url == "https://www.site.test/b"

    def test_a_redirect_to_a_loopback_address_is_refused(self, web_fetch):
        handler = web_fetch._SameSiteRedirectHandler()
        request = urllib.request.Request("https://site.test/a")
        with pytest.raises(web_fetch.FetchRefused):
            handler.redirect_request(request, None, 302, "Found", email.message.Message(),
                                     "http://127.0.0.1/b")

    def test_the_redirect_chain_is_bounded(self, web_fetch):
        assert web_fetch._SameSiteRedirectHandler.max_redirections <= 5


class TestTextExtraction:
    def test_script_and_style_content_never_reaches_the_agent(self, web_fetch, monkeypatch):
        _serve(monkeypatch, PAGE)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert "decoy" not in result["text"]
        assert "javascript" not in result["text"]
        assert "display: none" not in result["text"]

    def test_markup_is_dropped_and_entities_decoded(self, web_fetch, monkeypatch):
        _serve(monkeypatch, PAGE)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert "<b>" not in result["text"] and "<p>" not in result["text"]
        assert "Ada's hobby is rock climbing." in result["text"]
        assert result["title"] == "Ada & Friends"

    def test_whitespace_is_collapsed(self, web_fetch, monkeypatch):
        _serve(monkeypatch, b"<html><body><p>one   \t\t  two</p>\n\n\n" + b"<p>pad</p>" * 60 +
               b"</body></html>")
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert "one two" in result["text"]
        assert "\n\n" not in result["text"]
        assert "  " not in result["text"]

    def test_plain_text_is_returned_as_is(self, web_fetch, monkeypatch):
        _serve(monkeypatch, b"the answer is 41 plus one. " * 20, content_type="text/plain")
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert "the answer is 41 plus one." in result["text"]

    def test_binary_content_types_are_reported_not_parsed(self, web_fetch, monkeypatch):
        _serve(monkeypatch, b"\x89PNG\r\n\x1a\n" + b"\x00" * 500, content_type="image/png")
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert result["status"] == "unsupported_content_type"
        assert "image/png" in result["note"]


class TestJavaScriptRenderedPages:
    """The in-game guide warns the answer may not be in the HTML at all."""

    def test_an_empty_page_is_reported_rather_than_answered_from(self, web_fetch, monkeypatch):
        _serve(monkeypatch, EMPTY_PAGE)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert result["status"] == "no_readable_text"
        assert "javascript" in result["note"].lower()
        assert "another" in result["note"].lower()  # tells the agent what to do next
        assert result["text"].strip() == ""


class TestTokenEconomy:
    """The run bonus is 1000 - tokens/challenges, so the return stays small."""

    def test_max_chars_bounds_the_return(self, web_fetch, monkeypatch):
        _serve(monkeypatch, b"<html><body><p>" + b"long. " * 5000 + b"</p></body></html>")
        status, result = _call(web_fetch, url="https://site.test/p", max_chars=500)
        assert status == 200, result
        assert result["chars"] <= 500
        assert result["truncated"] is True

    def test_the_default_return_is_capped_without_being_asked(self, web_fetch, monkeypatch):
        _serve(monkeypatch, b"<html><body><p>" + b"long. " * 20000 + b"</p></body></html>")
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status == 200, result
        assert result["chars"] <= web_fetch.DEFAULT_MAX_CHARS

    def test_a_search_term_returns_only_the_passage_around_it(self, web_fetch, monkeypatch):
        page = (b"<html><body><p>" + b"noise. " * 2000 +
                b"the vault code is 8812. " + b"noise. " * 2000 + b"</p></body></html>")
        _serve(monkeypatch, page)
        status, result = _call(web_fetch, url="https://site.test/p", search="vault code")
        assert status == 200, result
        assert "the vault code is 8812." in result["text"]
        assert result["matches"] == 1
        assert result["chars"] < 2000  # not the whole page

    def test_a_search_term_that_is_absent_says_so(self, web_fetch, monkeypatch):
        _serve(monkeypatch, PAGE)
        status, result = _call(web_fetch, url="https://site.test/p", search="submarine")
        assert status == 200, result
        assert result["status"] == "no_match"
        assert result["matches"] == 0
        assert result["text"]  # the page is still returned to read

    def test_the_download_is_bounded(self, web_fetch, monkeypatch):
        """A huge page must not be pulled into memory whole."""
        captured = {}

        def _fake_urlopen(request, timeout=None):
            class _Huge(_FakeResponse):
                def read(self, size=None):
                    captured["size"] = size
                    return b"<html><body><p>x</p></body></html>"

            return _Huge(b"")

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        _call(web_fetch, url="https://site.test/p")
        assert captured["size"] is not None
        assert captured["size"] <= web_fetch.MAX_DOWNLOAD_BYTES + 1


class TestNetworkFailures:
    """A failure has to come back as JSON the agent can read, never as a stack."""

    def test_an_http_error_is_reported(self, web_fetch, monkeypatch):
        def _raise(request, timeout=None):
            raise urllib.error.HTTPError("https://site.test/p", 404, "Not Found", None, None)

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status != 200
        assert "404" in result["error"]

    def test_an_unreachable_host_is_reported(self, web_fetch, monkeypatch):
        def _raise(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status != 200
        assert "error" in result

    def test_a_timeout_is_reported(self, web_fetch, monkeypatch):
        def _raise(request, timeout=None):
            raise TimeoutError()

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        status, result = _call(web_fetch, url="https://site.test/p")
        assert status != 200
        assert "respond" in result["error"]


class TestDisqualificationRules:
    """Scanned from the source, because each of these ends the entry.

    From the official rules: a Lambda tool may not call Bedrock or any other
    model API, may not contact a site the challenge did not define, and may not
    hardcode answers; and no library may be installed beyond what is already
    there.
    """

    @pytest.fixture
    def source(self):
        with open(_TOOL_PATH, encoding="utf-8") as handle:
            return handle.read()

    def test_no_url_is_hardcoded(self, source):
        """Every fetched address must come from the caller."""
        assert "://" not in source

    def test_no_model_or_aws_api_is_called(self, source):
        lowered = _code_only(source).lower()
        for forbidden in ("boto3", "bedrock", "invoke_model", "openai", "anthropic",
                          "converse(", "sagemaker"):
            assert forbidden not in lowered, forbidden

    def test_only_the_standard_library_is_imported(self, source):
        """No library may be installed, so nothing outside the stdlib may be used."""
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
        for banned in ("requests", "bs4", "beautifulsoup4", "lxml", "httpx", "selenium"):
            assert banned not in imported

    def test_no_answer_table(self, source):
        """Nothing keyed by question text: the answer comes off the wire."""
        lowered = _code_only(source).lower()
        for smell in ("answers = {", "answer_table", "known_answers", "hobby"):
            assert smell not in lowered, smell
