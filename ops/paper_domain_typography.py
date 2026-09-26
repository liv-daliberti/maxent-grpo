"""Apply the paper's typewriter face to benchmark domain names in rendered TeX.

Use this only at TeX rendering boundaries: raw data, code, plots, and labels keep
ordinary domain strings. Existing literal/code regions and TeX identifiers are
preserved. ``exclude_phrases`` identifies literal prose about the Python language
(or other non-domain uses) that should retain its original typography.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

_DOMAIN = re.compile(r"(?<![\w\\])(?:PythonFactors|PantryPlan|Countdown|MathIR|Python|Graph|Pantry)(?!\w)|(?<![\w\\])Countd\.(?!\w)")
_COMMAND = re.compile(r"\\([A-Za-z@]+\*?|.)", re.S)
_LITERAL_ENVS = {"verbatim", "verbatim*", "Verbatim", "BVerbatim", "LVerbatim", "lstlisting", "minted", "alltt"}
_PROTECTED = {
    "texttt", "url", "path", "nolinkurl", "label", "ref", "eqref", "pageref",
    "autoref", "cref", "Cref", "vref", "nameref", "input", "include",
    "includegraphics", "lstinputlisting", "bibliography", "bibliographystyle",
    "addbibresource", "index", "gls", "Gls", "acrshort", "acrlong", "hypertarget",
}


def _argument_end(text: str, start: int, opening: str = "{", closing: str = "}") -> int:
    """Return one past a balanced TeX argument, retaining escaped delimiters."""
    depth, pos = 1, start + 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if not depth:
                return pos + 1
        pos += 1
    return len(text)


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def format_domain_names(tex: str, *, exclude_phrases: Iterable[str] = ()) -> str:
    """Wrap visible benchmark names in ``\\texttt{}``, without nesting wrappers.

    Excluded phrases are exact literal strings; their characters remain untouched.
    Citations, reference/asset identifiers, comments, and literal/code text are
    skipped. Generic language mentions should be supplied in ``exclude_phrases``.
    """
    protected = bytearray(len(tex))
    for phrase in exclude_phrases:
        if not phrase:
            continue
        start = 0
        while (start := tex.find(phrase, start)) != -1:
            protected[start:start + len(phrase)] = b"\1" * len(phrase)
            start += len(phrase)
    pos = 0
    while pos < len(tex):
        start = pos
        if tex[pos] == "%":
            stop = tex.find("\n", pos)
            pos = len(tex) if stop < 0 else stop
            protected[start:pos] = b"\1" * (pos - start)
            continue
        if tex[pos] != "\\":
            pos += 1
            continue
        command = _COMMAND.match(tex, pos)
        if not command:
            pos += 1
            continue
        name = command.group(1).rstrip("*")
        pos = command.end()
        protected[start:pos] = b"\1" * (pos - start)
        if name in {"verb", "lstinline"}:
            if name == "lstinline":
                pos = _skip_space(tex, pos)
                if pos < len(tex) and tex[pos] == "[":
                    pos = _argument_end(tex, pos, "[", "]")
            if pos < len(tex):
                delimiter = tex[pos]
                if delimiter == "{":
                    pos = _argument_end(tex, pos)
                else:
                    stop = tex.find(delimiter, pos + 1)
                    pos = len(tex) if stop < 0 else stop + 1
                protected[start:pos] = b"\1" * (pos - start)
            continue
        arg = _skip_space(tex, pos)
        if name == "begin" and arg < len(tex) and tex[arg] == "{":
            end = _argument_end(tex, arg)
            environment = tex[arg + 1:end - 1]
            if environment in _LITERAL_ENVS:
                closing = "\\end{" + environment + "}"
                stop = tex.find(closing, end)
                pos = len(tex) if stop < 0 else stop + len(closing)
                protected[start:pos] = b"\1" * (pos - start)
                continue
        skip = name in _PROTECTED or name.lower().startswith("cite") or name in {"href", "hyperref", "begin", "end"}
        if skip:
            while arg < len(tex) and tex[arg] == "[":
                arg = _skip_space(tex, _argument_end(tex, arg, "[", "]"))
            # A hyperref's optional argument is its identifier; its braced text
            # remains visible. A href's first braced argument is its URL.
            if name != "hyperref" and arg < len(tex) and tex[arg] == "{":
                pos = _argument_end(tex, arg)
            else:
                pos = arg
            protected[start:pos] = b"\1" * (pos - start)
    parts, previous = [], 0
    for match in _DOMAIN.finditer(tex):
        if any(protected[match.start():match.end()]):
            continue
        parts.extend((tex[previous:match.start()], "\\texttt{" + match.group() + "}"))
        previous = match.end()
    parts.append(tex[previous:])
    return "".join(parts)
