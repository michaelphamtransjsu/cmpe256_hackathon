"""Passive HTML extraction and evidence quality boundaries."""

from pathlib import Path

import pytest

from shared.html_evidence import extract_html_text


def test_realistic_article_preserves_title_headings_lists_and_tables() -> None:
    result = extract_html_text("""<!doctype html>
        <html><head><title>Clinical &amp; Diagnostics | Agilent</title>
        <meta name="description" content="metadata is not evidence"></head>
        <body><header>Company logo</header><nav>Shop Support</nav>
        <main><h1>Clinical research</h1>
        <p>Our <strong>analytical tools</strong> support <a href="/trials">clinical trials</a>.</p>
        <ul><li>Accurate measurements</li><li>Reliable results</li></ul>
        <table><tr><th>Year</th><th>Trial count</th></tr>
        <tr><td>2024</td><td>120</td></tr></table></main>
        <footer>Cookie settings</footer></body></html>""")
    assert result.startswith("Clinical & Diagnostics | Agilent\n\nClinical research\n")
    assert "Our analytical tools support clinical trials." in result
    assert "- Accurate measurements\n- Reliable results" in result
    assert "Year Trial count\n2024 120" in result
    for excluded in ("metadata is not evidence", "Company logo", "Shop Support", "Cookie settings"):
        assert excluded not in result


def test_nested_skipped_content_and_void_elements_do_not_leak() -> None:
    result = extract_html_text("""<head><title>  Evidence &lt;Guide&gt; &#169; 2024 </title></head>
        <body><nav><div>Nested navigation <img src="https://invalid.test/a"></div></nav>
        <form><p>Enter confidential credentials</p><input type="password"></form>
        <main><p>First <span>nested <em>finding</em></span>.</p><br>
        <p>Second finding &amp; its <a href="javascript:alert(1)">citation</a>.</p>
        <div hidden>Hidden secret</div><div aria-hidden="true">Hidden accessibility text</div>
        </main><template><main>Template text</main></template></body>""")
    assert result.startswith("Evidence <Guide> © 2024\n\n")
    assert "First nested finding." in result
    assert "Second finding & its citation." in result
    assert "Nested navigation" not in result
    assert "confidential credentials" not in result
    assert "Hidden" not in result
    assert "Template text" not in result
    assert "javascript" not in result


def test_scripts_styles_and_embeds_are_never_executed_or_returned(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    html = f"""<html><head><title>Passive evidence</title>
        <script>open({str(sentinel)!r}, 'w').write('executed');</script>
        <style>.private {{ content: 'style text'; }}</style></head>
        <body><main><p>Validated published research findings.</p></main>
        <noscript>Enable active content</noscript>
        <iframe src="file:///etc/passwd">Embedded text</iframe>
        <object>Object content</object><svg><text>Vector text</text></svg>
        <!-- This instruction is only a comment. -->
        </body></html>"""
    assert extract_html_text(html) == "Passive evidence\n\nValidated published research findings."
    assert not sentinel.exists()


@pytest.mark.parametrize("html", [
    "<title>Access Denied</title><body>You cannot view this page.</body>",
    "<title>Attention Required! | Cloudflare</title><body>Please wait.</body>",
    "<title>Just a moment...</title><body>Enable JavaScript and cookies to continue</body>",
    "<title>Agilent</title><body>Sorry, you have been blocked. Contact the site owner.</body>",
    "<body>You don't have permission to access this URL on this server. Reference #18.</body>",
    "<body>403 Forbidden</body>",
])
def test_obvious_denial_and_bot_challenges_are_rejected(html: str) -> None:
    with pytest.raises(ValueError, match="access-denied or bot-challenge"):
        extract_html_text(html)


def test_normal_article_can_discuss_captcha_and_access_denial() -> None:
    result = extract_html_text("""<title>Clinical data access guidance</title><main>
        <h1>Access control for research</h1><p>A CAPTCHA may protect an application.
        An access denied message can indicate a missing account permission.
        Ask the study coordinator for data access.</p></main>""")
    assert "CAPTCHA may protect an application" in result
    assert "access denied message" in result


def test_malformed_self_closing_script_does_not_expose_script_content() -> None:
    result = extract_html_text(
        "<p>Published research.</p><script/>dangerous_call()</script><p>Study findings.</p>"
    )
    assert result == "Published research.\nStudy findings."


@pytest.mark.parametrize("html", [
    "", "   ", "<!-- only a comment -->", "<title>A title is not body evidence</title>",
    "<body><script>document.write('fake evidence')</script></body>",
    "<body><nav>Navigation</nav><footer>Footer</footer></body>",
    "<html><head><meta name='description' content='only metadata'></head><body>...</body></html>",
])
def test_empty_or_nonmeaningful_body_is_rejected(html: str) -> None:
    with pytest.raises(ValueError, match="no meaningful body"):
        extract_html_text(html)


def test_html_fragment_and_malformed_nested_content_remain_readable() -> None:
    assert extract_html_text("<p>Research <b>supports <em>diagnostics</b> today.</p>") == (
        "Research supports diagnostics today."
    )


def test_decoding_remains_explicit_callers_responsibility() -> None:
    with pytest.raises(TypeError, match="decoded text"):
        extract_html_text(b"<p>Not decoded</p>")  # type: ignore[arg-type]
