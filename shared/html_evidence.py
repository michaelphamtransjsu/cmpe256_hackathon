"""Extract passive, readable evidence from locally supplied HTML.

This module neither fetches resources nor evaluates markup, scripts, or document
instructions.  It deliberately omits site navigation and interactive content so
that a retrieved page contributes article text rather than executable markup.
"""

from __future__ import annotations

from html.parser import HTMLParser
import re


_DISCARDED = frozenset({
    "script", "style", "noscript", "template", "nav", "footer", "header",
    "form", "iframe", "object", "embed", "svg", "canvas",
})
_VOID = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})
_BLOCK = frozenset({
    "address", "article", "aside", "blockquote", "dd", "details", "dialog",
    "div", "dl", "dt", "fieldset", "figcaption", "figure", "h1", "h2",
    "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p", "pre",
    "section", "summary", "table", "tr", "ul",
})
_CHALLENGE_TITLES = re.compile(
    r"^(?:access denied|403(?: forbidden)?|forbidden|request blocked|"
    r"attention required(?:\s*[!|:-].*)?|just a moment[.!\s]*|"
    r"checking your browser(?:\s.*)?|verify (?:that )?you are human|"
    r"security verification|robot challenge)$",
    re.IGNORECASE,
)
_CHALLENGE_BODY_START = re.compile(
    r"^(?:sorry,? you have been blocked|"
    r"checking your browser before accessing|"
    r"(?:please )?enable javascript and cookies to continue|"
    r"please wait while your request is being verified|"
    r"you (?:do not|don't) have permission to access .+ on this server)",
    re.IGNORECASE,
)


def _clean_text(parts: list[str]) -> str:
    """Keep structural lines while folding source indentation and entities."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parts).splitlines()]
    return "\n".join(line for line in lines if line)


class _EvidenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Each stack item stores its inherited suppression state.  Matching
        # closing tags unwind malformed nesting without making skipped data live.
        self.stack: list[tuple[str, bool]] = []
        self.body: list[str] = []
        self.title: list[str] = []

    @property
    def suppressed(self) -> bool:
        return bool(self.stack and self.stack[-1][1])

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        hidden = any(
            key == "hidden" or (key == "aria-hidden" and (value or "").lower() == "true")
            for key, value in attrs
        )
        suppressed = self.suppressed or tag in _DISCARDED or hidden
        in_head = any(name == "head" for name, _ in self.stack)
        if not suppressed and not in_head and tag != "title":
            if tag in _BLOCK or tag == "br":
                self.body.append("\n")
            if tag == "li":
                self.body.append("- ")
            if tag in {"td", "th"}:
                self.body.append("\t")
        if tag not in _VOID:
            self.stack.append((tag, suppressed))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        # In HTML, a slash does not close non-void script/style elements.  Keep
        # their following data suppressed even for malformed injection markup.
        if tag not in _VOID and tag not in {"script", "style"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.suppressed and tag in _BLOCK:
            self.body.append("\n")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.suppressed:
            return
        names = {tag for tag, _ in self.stack}
        if "title" in names:
            self.title.append(data)
        elif "head" not in names:
            self.body.append(data)


def extract_html_text(html: str) -> str:
    """Return a page title and readable body, or reject unusable evidence.

    The input must already be downloaded and decoded by the caller.  Obvious
    access-denied and bot-challenge pages are rejected instead of indexed as
    evidence.  A normal page mentioning CAPTCHA or access control remains valid.
    """
    if not isinstance(html, str):
        raise TypeError("HTML evidence must be decoded text")
    parser = _EvidenceParser()
    parser.feed(html)
    parser.close()
    title = re.sub(r"\s+", " ", "".join(parser.title)).strip()
    body = _clean_text(parser.body)
    flat_body = re.sub(r"\s+", " ", body).strip()
    if _CHALLENGE_TITLES.fullmatch(title) or (
        len(flat_body) < 2000 and (
            _CHALLENGE_BODY_START.match(flat_body)
            or flat_body.lower() in {"access denied", "403 forbidden", "forbidden"}
        )
    ):
        raise ValueError("HTML evidence is an access-denied or bot-challenge page")
    if not any(character.isalnum() for character in body):
        raise ValueError("HTML evidence contains no meaningful body text")
    if title and title.casefold() != body.casefold():
        return f"{title}\n\n{body}"
    return body
