"""Text normalisation.

Every hash, every lexicon lookup, and every skill comparison in the system goes
through here. That is deliberate: the Appendix A.3 case where a headline gains a
rocket emoji and a double space has to resolve to ``materiality: noise`` with no
special-case branch anywhere in the delta engine. It resolves to noise because
the normalised forms are byte-identical, and that only holds if exactly one
normaliser is in play.
"""

from __future__ import annotations

import re
import unicodedata

# Variation selectors, ZWJ/ZWNJ, skin-tone modifiers, regional indicators.
_ZERO_WIDTH = re.compile(
    "[​-‏⁠﻿︀-️\U0001F3FB-\U0001F3FF\U0001F1E6-\U0001F1FF]"
)
_WHITESPACE = re.compile(r"\s+")
_LEGAL_SUFFIX = re.compile(
    r"\b(pvt\.?|private|ltd\.?|limited|inc\.?|incorporated|llp|llc|corp\.?|"
    r"corporation|co\.?|gmbh|technologies|technology|labs|solutions|systems|"
    r"services|india)\b"
)
_PUNCT_EDGES = re.compile(r"^[^\w]+|[^\w]+$")

# ".js" style suffixes that make React / React.js / Reactjs three tokens.
_JS_SUFFIX = re.compile(r"(?:\.js|\s+js|-js|js)$")


def _is_symbol_or_pictograph(ch: str) -> bool:
    """True for emoji and other pictographic symbols, false for real punctuation."""
    if unicodedata.category(ch) != "So":
        return False
    return True


def strip_emoji(s: str) -> str:
    """Remove pictographs, zero-width joiners and variation selectors."""
    s = _ZERO_WIDTH.sub("", s)
    return "".join(ch for ch in s if not _is_symbol_or_pictograph(ch))


def norm_text(s: str | None) -> str:
    """NFKC -> strip emoji -> collapse whitespace -> strip -> casefold."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_emoji(s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip().casefold()


def norm_company(s: str | None) -> str:
    """Normalise a company name so ``Zeta`` and ``Zeta Pvt Ltd`` are one key."""
    text = norm_text(s)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = _LEGAL_SUFFIX.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def norm_skill(s: str | None, aliases: dict[str, str] | None = None) -> str:
    """Normalise one skill string to its canonical form.

    ``aliases`` maps an already-normalised variant to its canonical name, e.g.
    ``{"reactjs": "react", "k8s": "kubernetes"}``. It is passed in rather than
    imported so that ``core`` never reads a file.
    """
    text = norm_text(s)
    if not text:
        return ""
    text = _PUNCT_EDGES.sub("", text)
    text = text.replace("_", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    if aliases:
        # Alias lookup happens before suffix stripping so an alias can pin an
        # exception (e.g. "node.js" must stay "node.js", not become "node").
        hit = aliases.get(text)
        if hit:
            return hit
    stripped = _JS_SUFFIX.sub("", text).strip()
    if stripped and stripped != text:
        if aliases and stripped in aliases:
            return aliases[stripped]
        text = stripped
    if aliases:
        text = aliases.get(text, text)
    return text


#: Seniority-ish words that a title classifier should not treat as the role itself.
SENIORITY_PREFIXES = {
    "senior", "sr", "sr.", "junior", "jr", "jr.", "lead", "principal", "staff",
    "associate", "assistant", "chief", "head", "deputy", "trainee", "intern",
    "graduate", "entry", "level", "i", "ii", "iii", "iv",
}


def norm_title(s: str | None) -> tuple[str, list[str]]:
    """Split a raw title into ``(title_core, seniority_prefix_tokens)``.

    Handles the ``Role @ Company`` and ``Role at Company`` shapes that show up in
    LinkedIn headlines, and drops the pipe-delimited tail that Indian-market
    headlines use for a skills list (``SDE-3 | Python | AWS | Ex-Amazon``).
    """
    text = norm_text(s)
    if not text:
        return "", []
    # A headline is often "title @ company | skill | skill". Keep the head.
    text = text.split("|")[0]
    text = re.split(r"\s+@\s+|\s+@|@\s+|\s+at\s+", text)[0]
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w+#/. -]", " ", text)
    text = _WHITESPACE.sub(" ", text).strip(" -.")

    # "sde-3", "sde3", "swe-ii" -> core "sde"/"swe" plus a level token. Indian
    # job titles carry the level welded onto the acronym far more often than the
    # spelled-out form does.
    text = re.sub(r"\b(sde|swe|mts|sse)[-\s]?([1-4]|i{1,3}|iv)\b", r"\1 \2", text)

    tokens = text.split()
    prefixes: list[str] = []
    while tokens and tokens[0] in SENIORITY_PREFIXES:
        prefixes.append(tokens.pop(0))
    # Trailing level markers: "software engineer ii", "sde iii".
    while tokens and tokens[-1] in {"i", "ii", "iii", "iv", "1", "2", "3", "4"}:
        prefixes.append(tokens.pop())
    core = " ".join(tokens) if tokens else text
    return core.strip(), prefixes


def tokens(s: str | None) -> list[str]:
    """Word tokens of the normalised form. Used for evidence matching."""
    return re.findall(r"[a-z0-9+#.]+", norm_text(s))
