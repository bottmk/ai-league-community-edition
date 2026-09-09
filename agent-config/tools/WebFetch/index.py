# WebFetch Tool - AI League
# =========================
# Retrieves a URL the caller supplies and returns its readable text.
#
# Exists for the challenge tiles that quote a page and ask for a fact printed
# on it (c4 web scraping, c23 web fetch, and the c6 boss, which needs a fetch
# and then a calculation). A language model cannot answer those from its own
# weights, so the fact has to come off the wire.
#
# Rules this tool is shaped by, from the official AI League rules page:
#   - No Bedrock, no OpenAI, no model API of any kind may be called from a
#     Lambda tool. Nothing here calls a model; it moves bytes and strips tags.
#   - Calling an external site the challenge did not name is a disqualification,
#     so there is NO hardcoded URL anywhere in this file, no crawling, and no
#     following a redirect that leaves the site the caller asked for. Only the
#     caller's URL, plus same-site redirect hops, is ever contacted.
#   - Hardcoding answers is a disqualification, so there is no lookup table
#     here either: every answer comes out of the fetched bytes.
#   - No extra libraries may be installed. This is Python standard library
#     only: urllib.request for the fetch, html.parser for the markup.
#
# Token economy: the run bonus is 1000 - (total tokens / challenges visited),
# so returning a whole page is expensive. The default return is capped, and a
# 'search' term narrows the return to the passages around that term, which is
# what a small model actually needs to read one fact off a page.
#
# Deployed as the AgentCoreGatewayTool-<name> Lambda behind the AgentCore
# Gateway. The gateway hands the tool arguments in flat, at the top level of
# the event ({"url": "...", "search": "..."}), and expects
# {"statusCode": 200, "body": json.dumps(result)} back. The 'body' unwrapping
# below is a fallback for an API Gateway style invocation.
#
# In the AI League environment this file is named lambda_function.py, so the
# handler setting reads lambda_function.lambda_handler. The Lambda times out at
# 30 seconds; the network timeouts below keep a slow host well inside that.

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# Identifies the fetcher without advertising a site. Deliberately contains no
# URL: a hardcoded address anywhere in a tool is a disqualification risk.
USER_AGENT = "AILeagueWebFetch/1.0 (Python urllib; stdlib only)"

CONNECT_TIMEOUT = 8.0      # seconds per socket operation
MAX_REDIRECTS = 5          # same-site hops only, see _SameSiteRedirectHandler
MAX_DOWNLOAD_BYTES = 2_000_000

DEFAULT_MAX_CHARS = 4000
MIN_MAX_CHARS = 200
LIMIT_MAX_CHARS = 20000

SNIPPET_RADIUS = 300       # characters kept either side of a search hit
MAX_SNIPPETS = 6
SNIPPET_JOIN = "\n...\n"

# Under this many characters of readable text the page is reported as empty
# rather than answered from. The in-game guide warns that a challenge page may
# be rendered by JavaScript, in which case the HTML holds no answer at all and
# the agent must be told so instead of guessing from nothing.
MIN_READABLE_CHARS = 200

ALLOWED_SCHEMES = ("http", "https")

# Hosts that never leave the machine. Checked by name before any DNS lookup so
# that no packet is sent for them at all.
BLOCKED_HOST_NAMES = ("localhost", "localhost.localdomain")

# Markup whose text content is code or styling, never prose.
SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "canvas", "math",
    "object", "embed", "iframe",
})

# Markup that ends a line of prose.
BLOCK_TAGS = frozenset({
    "p", "div", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd", "tr", "table",
    "thead", "tbody", "section", "article", "header", "footer", "nav", "aside",
    "main", "form", "figure", "figcaption", "blockquote", "pre", "address",
    "h1", "h2", "h3", "h4", "h5", "h6",
})

# Content types worth turning into text. Anything else is reported, not parsed.
TEXTUAL_CONTENT_HINTS = ("text/", "json", "xml", "html", "javascript", "csv")


class FetchRefused(Exception):
    """The URL, or a redirect hop, is one this tool must not contact."""


def lambda_handler(event, context):
    """
    AWS Lambda function that fetches a web page and returns its readable text.
    Handles both API Gateway format and direct AgentCore Gateway format.

    ---
    Tool: web_fetch
    Description: Fetches the URL given in the question and returns the page's readable text, with scripts, styles and markup removed. Use it whenever a question quotes a URL or asks what a page says.
    Parameters:
        url        (required) - the http or https URL from the question, fetched exactly as given
        search     (optional) - a word or phrase to look for. Only the passages around it are returned, which is much cheaper than the whole page
        max_chars  (optional) - how many characters of text to return. Defaults to 4000, maximum 20000
    ---

    ## Return
        url       - the URL actually read, after any same-site redirect
        status    - ok, no_match, no_readable_text, or unsupported_content_type
        title     - the page title, empty when it has none
        text      - the readable text, or the matching passages when search was given
        chars     - length of the returned text
        truncated - true when the page held more text than was returned
        note      - present only when something needs saying, e.g. the page is
                    rendered by JavaScript and the fetched HTML holds no answer
        search    - echoed back when a search term was given
        matches   - number of times the search term appears in the page

    A refused URL, an unreachable host or an HTTP error returns a non-200 with
    an 'error' key. Only http and https are fetched; private, loopback and
    link-local addresses are refused, and a redirect that leaves the requested
    site is refused rather than followed.
    """

    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        url = _as_url_text(body.get('url'))
        if not url:
            return _err(400, 'Missing required parameter: url')

        search = body.get('search') or body.get('query') or ''
        search = str(search).strip()
        max_chars = _as_int(body.get('max_chars'), DEFAULT_MAX_CHARS)
        max_chars = max(MIN_MAX_CHARS, min(LIMIT_MAX_CHARS, max_chars))

        print(f"DEBUG: fetching url={url} search={search!r} max_chars={max_chars}")

        try:
            _check_url(url)
        except FetchRefused as refused:
            return _err(400, str(refused))

        try:
            raw, final_url, content_type, over_size = _fetch(url)
        except FetchRefused as refused:
            return _err(400, str(refused))
        except urllib.error.HTTPError as http_error:
            return _err(502, f'HTTP {http_error.code} from the page: {http_error.reason}')
        except urllib.error.URLError as url_error:
            return _err(502, f'Could not reach the page: {url_error.reason}')
        except (socket.timeout, TimeoutError):
            return _err(504, f'The page did not respond within {CONNECT_TIMEOUT:g} seconds')

        result = _build_result(raw, final_url, content_type, over_size, search, max_chars)
        print(f"RESULT: status={result['status']} chars={result['chars']}")
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:  # noqa: BLE001 - the gateway needs a JSON body, never a stack
        print(f"ERROR: {e}")
        return _err(500, str(e))


def _err(code, msg):
    return {'statusCode': code, 'body': json.dumps({'error': msg})}


# ---------------------------------------------------------------------------
# Parameter parsing
#
# A language model fills these in, so every value may arrive as a string, and
# a URL may arrive wrapped in quotes, angle brackets or trailing punctuation.
# ---------------------------------------------------------------------------


def _as_int(value, fallback):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _as_url_text(value):
    if value is None:
        return ''
    text = str(value).strip()
    text = text.strip('<>"“”‘’\' \t\n\r')
    text = text.rstrip('.,;')
    return text


# ---------------------------------------------------------------------------
# Destination vetting
#
# The tool contacts the caller's host and nothing else. Everything below exists
# to keep that true: no other scheme, no address that lives inside the network
# the Lambda runs in, and no redirect that walks off the requested site.
# ---------------------------------------------------------------------------


def _resolve_addresses(host):
    """Every IP the host resolves to. Split out so it can be stubbed in tests."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


def _blocked_address(text):
    """True when an IP literal points back into private or local space."""
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if getattr(address, 'ipv4_mapped', None):
        address = address.ipv4_mapped
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _check_url(url):
    """Raise FetchRefused unless this is a public http(s) URL. Returns the host."""
    parts = urllib.parse.urlsplit(url)
    scheme = (parts.scheme or '').lower()
    if scheme not in ALLOWED_SCHEMES:
        return _refuse(
            'Only http and https URLs are fetched, not '
            + (scheme or 'a URL with no scheme')
        )

    try:
        host = parts.hostname
    except ValueError as bad_host:
        return _refuse(f'Malformed host in the URL: {bad_host}')
    if not host:
        return _refuse('The URL has no host')

    host = host.lower().rstrip('.')
    if host in BLOCKED_HOST_NAMES or host.endswith('.localhost'):
        return _refuse(f'Refusing to fetch a local address: {host}')

    if _blocked_address(host):
        return _refuse(f'Refusing to fetch a private, loopback or link-local address: {host}')

    try:
        addresses = _resolve_addresses(host)
    except socket.gaierror as unresolved:
        return _refuse(f'Could not resolve host {host}: {unresolved}')
    if not addresses:
        return _refuse(f'Could not resolve host {host}')
    for address in addresses:
        if _blocked_address(address):
            return _refuse(
                f'Refusing to fetch {host}: it resolves to the private or local address {address}'
            )
    return host


def _refuse(message):
    raise FetchRefused(message)


def _same_site(current_url, next_url):
    """True when a redirect stays on the site the caller asked for.

    A host is the same site as itself and as any parent or subdomain of itself,
    which covers the ordinary bare-domain to www and http to https hops. A hop
    to an unrelated domain is a different site and is refused: contacting a site
    the challenge did not name is a disqualification.
    """
    current = (urllib.parse.urlsplit(current_url).hostname or '').lower().rstrip('.')
    following = (urllib.parse.urlsplit(next_url).hostname or '').lower().rstrip('.')
    if not current or not following:
        return False
    if current == following:
        return True
    return current.endswith('.' + following) or following.endswith('.' + current)


class _SameSiteRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Vets every redirect hop before urllib follows it."""

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        if not _same_site(req.full_url, newurl):
            _refuse(
                f'The page redirects off-site to {newurl}. Refusing to follow it; '
                f'pass that URL explicitly if the question points there.'
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _urlopen(request, timeout):
    """Open a request through urlopen with the same-site redirect handler in place.

    The handler is installed only for the duration of the call, so the rest of
    the process keeps urllib's default behaviour, and the call still goes
    through urllib.request.urlopen, which keeps it stubbable in tests.
    """
    previous = getattr(urllib.request, '_opener', None)
    urllib.request.install_opener(urllib.request.build_opener(_SameSiteRedirectHandler))
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    finally:
        urllib.request._opener = previous


def _fetch(url):
    """Download the caller's URL. Returns (bytes, final_url, content_type, over_size)."""
    request = urllib.request.Request(url, headers={
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5',
        'Accept-Language': 'en,ja;q=0.9,*;q=0.5',
    })
    response = _urlopen(request, CONNECT_TIMEOUT)
    try:
        # One byte past the cap tells us the page was larger without holding it.
        raw = response.read(MAX_DOWNLOAD_BYTES + 1) or b''
        headers = getattr(response, 'headers', None)
        if headers is None and hasattr(response, 'info'):
            headers = response.info()
        content_type = ''
        if headers is not None:
            try:
                content_type = headers.get('Content-Type', '') or ''
            except AttributeError:
                content_type = ''
        final_url = url
        if hasattr(response, 'geturl'):
            final_url = response.geturl() or url
    finally:
        closer = getattr(response, 'close', None)
        if callable(closer):
            closer()

    over_size = len(raw) > MAX_DOWNLOAD_BYTES
    return raw[:MAX_DOWNLOAD_BYTES], final_url, content_type, over_size


# ---------------------------------------------------------------------------
# Markup to text
# ---------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Collects prose, dropping script, style and every tag.

    convert_charrefs is left on, so entities are decoded for us.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ''
        self._parts = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == 'title':
            self._in_title = True
        elif tag in BLOCK_TAGS:
            self._parts.append('\n')
        elif tag in ('td', 'th'):
            self._parts.append(' ')

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == 'title':
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self._parts.append('\n')

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        else:
            self._parts.append(data)

    def get_text(self):
        return ''.join(self._parts)


def _decode(raw, content_type):
    """Bytes to text, honouring the declared charset and never raising."""
    charset = ''
    match = re.search(r'charset=["\']?([\w.:-]+)', content_type or '', re.IGNORECASE)
    if match:
        charset = match.group(1)
    if not charset:
        head = raw[:4096]
        meta = re.search(rb'charset=["\']?([\w.:-]+)', head, re.IGNORECASE)
        if meta:
            charset = meta.group(1).decode('ascii', 'ignore')
    for candidate in (charset, 'utf-8', 'cp1252'):
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode('utf-8', 'replace')


def _collapse(text):
    """Collapse runs of whitespace so the return costs as few tokens as possible."""
    text = text.replace(' ', ' ')
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line)


def _is_textual(content_type):
    if not content_type:
        return True  # no declaration: try to read it rather than give up
    lowered = content_type.lower()
    return any(hint in lowered for hint in TEXTUAL_CONTENT_HINTS)


def _extract(raw, content_type):
    """Readable text and title for a downloaded body."""
    text = _decode(raw, content_type)
    lowered = (content_type or '').lower()
    if 'html' in lowered or 'xml' in lowered or '<' in text[:2048]:
        parser = _TextExtractor()
        try:
            parser.feed(text)
            parser.close()
        except Exception as parse_error:  # noqa: BLE001 - malformed markup is normal
            print(f"DEBUG: html parse stopped early: {parse_error}")
        return _collapse(parser.get_text()), _collapse(parser.title)
    return _collapse(text), ''


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _snippets(text, term, max_chars):
    """Passages around each occurrence of term, capped at max_chars.

    Returns (snippet_text, match_count). Returning only the neighbourhood of
    the term is what keeps the token bonus intact on a long page.
    """
    lowered_text = text.lower()
    lowered_term = term.lower()
    positions = []
    start = lowered_text.find(lowered_term)
    while start != -1 and len(positions) < MAX_SNIPPETS:
        positions.append(start)
        start = lowered_text.find(lowered_term, start + max(1, len(lowered_term)))
    matches = lowered_text.count(lowered_term)
    if not positions:
        return '', matches

    spans = []
    for position in positions:
        begin = max(0, position - SNIPPET_RADIUS)
        end = min(len(text), position + len(term) + SNIPPET_RADIUS)
        if spans and begin <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((begin, end))

    pieces = [text[begin:end].strip() for begin, end in spans]
    return SNIPPET_JOIN.join(pieces)[:max_chars], matches


def _build_result(raw, final_url, content_type, over_size, search, max_chars):
    result = {
        'url': final_url,
        'status': 'ok',
        'title': '',
        'text': '',
        'chars': 0,
        'truncated': False,
    }

    if not _is_textual(content_type):
        result['status'] = 'unsupported_content_type'
        result['note'] = (
            f'The URL returned {content_type.strip()}, which holds no readable text. '
            f'Ask for a different page.'
        )
        return result

    text, title = _extract(raw, content_type)
    result['title'] = title[:200]

    if len(text) < MIN_READABLE_CHARS:
        result['status'] = 'no_readable_text'
        result['text'] = text
        result['chars'] = len(text)
        result['note'] = (
            f'The fetched HTML yielded only {len(text)} characters of readable text. '
            f'The page is probably rendered by JavaScript, so the answer is not in the '
            f'HTML at all. Do not guess from this; try another URL or another source.'
        )
        return result

    if search:
        result['search'] = search
        snippet, matches = _snippets(text, search, max_chars)
        result['matches'] = matches
        if snippet:
            result['text'] = snippet
            result['chars'] = len(snippet)
            result['truncated'] = over_size or len(text) > len(snippet)
            return result
        result['status'] = 'no_match'
        result['note'] = (
            f'"{search}" does not appear on the page. The start of the page is returned '
            f'instead; search for a different word or read the whole page.'
        )

    body = text[:max_chars]
    result['text'] = body
    result['chars'] = len(body)
    result['truncated'] = over_size or len(body) < len(text)
    return result
