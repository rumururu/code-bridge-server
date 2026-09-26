"""Reading a provider's failure text well enough to know it is not a defect.

Every provider funnels a failure into the same untyped shape — an
``{"type": "error", "error": {"message": ...}}`` event (the contract in
:mod:`llm.llm_session`) whose message is whatever the CLI printed. That string
is faithfully carried all the way to the browser and shown, and until now
nothing ever looked at it.

That is fine for most failures and wrong for exactly one: **"this account is
out of allowance"**. A bad request, an unknown model, a missing binary and a
crash are all *this did not work* — the honest thing is to show them and stop.
Being out of quota is *this backend cannot answer for you today, and another
installed one can*, which is a different sentence and deserves a different
screen.

So this module answers one narrow question, and answers ``None`` whenever it is
not sure:

    is this message the provider saying it is out of allowance?

**Under-matching is the safe direction.** A message wrongly read as a quota
problem invites the user to switch providers to fix a defect that will follow
them to the next one — hiding a real bug behind a shrug. A quota message
wrongly read as unknown just shows the raw text, which is what happened before
this module existed. Every pattern below is therefore a phrase a provider only
prints when an allowance is exhausted, never a generic word like ``error`` or
a bare status code: ``429`` alone is not enough, because it also shows up in
retry logs and stack traces that have nothing to do with the caller's account.

Observed samples the patterns are written against:

* Codex CLI — ``You've hit your usage limit. Upgrade to Plus to continue using
  Codex (https://chatgpt.com/explore/plus), or try again at Sep 15th, 2026``
* Claude — ``Claude usage limit reached. Your limit will reset at ...`` and
  ``Your credit balance is too low to access the Anthropic API``
* Gemini — ``429 Resource has been exhausted`` / ``RESOURCE_EXHAUSTED`` /
  ``Quota exceeded for quota metric ...``
"""

from __future__ import annotations

import re

# The one kind this module claims to recognise. Kept as a constant so the
# route, the wire format and the tests all spell it the same way.
PROVIDER_ERROR_QUOTA = "quota"

# Phrases that only appear when a provider is refusing on allowance grounds.
# Matched case-insensitively against the whole message.
_QUOTA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"usage limit",
        r"\bquota\b",
        r"insufficient_quota",
        r"resource[ _]has been exhausted",
        r"resource_exhausted",
        r"rate[ _]?limit",
        r"too many requests",
        r"credit balance is too low",
        r"upgrade to (plus|pro)\b",
        r"out of credits",
        r"exceeded your current",
    )
)

# Checked first. A message that says a provider is not installed, or that the
# request itself was malformed, must never be read as a quota problem even if
# it happens to contain one of the phrases above (a validation error quoting a
# field named `quota`, say). These are defects to surface, not backends to
# switch away from.
_NOT_QUOTA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"command not found",
        r"not installed",
        r"no such file",
        r"is not recognized as",
    )
)


def classify_provider_error(message: str | None) -> str | None:
    """Return ``"quota"`` for an exhausted-allowance message, else ``None``.

    ``None`` means "no claim made" — not "this is fine". Callers must keep
    reporting the failure either way; the classification only decides whether
    an *additional* offer (switch to another installed provider) makes sense.
    """
    if not message:
        return None
    text = str(message)
    if any(pattern.search(text) for pattern in _NOT_QUOTA_PATTERNS):
        return None
    if any(pattern.search(text) for pattern in _QUOTA_PATTERNS):
        return PROVIDER_ERROR_QUOTA
    return None
