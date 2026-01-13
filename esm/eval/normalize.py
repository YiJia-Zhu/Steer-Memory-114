from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def normalize_text_basic(s: str) -> str:
    return s.strip()


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+$", "", s)  # drop trailing punctuation
    return s


def normalize_number_str(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    # normalize unicode minus
    s = s.replace("−", "-")
    try:
        d = Decimal(s)
        # If it's integral, prefer a plain integer string (avoid '2.6E+2').
        if d == d.to_integral():
            return str(int(d))

        # Canonical decimal form: drop trailing zeros, avoid scientific notation.
        n = d.normalize()
        out = format(n, "f").rstrip("0").rstrip(".")
        return out if out else "0"
    except (InvalidOperation, ValueError):
        return s


_RE_LATEX_INLINE_MATH = re.compile(r"^\s*\$+(.*?)\$+\s*$", re.DOTALL)
_RE_LATEX_PARENS_MATH = re.compile(r"^\s*\\\((.*?)\\\)\s*$", re.DOTALL)
_RE_LATEX_BRACKETS_MATH = re.compile(r"^\s*\\\[(.*?)\\\]\s*$", re.DOTALL)


def _extract_balanced_braces(s: str, open_idx: int) -> tuple[str, int] | None:
    if open_idx < 0 or open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    start = None
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i], i + 1
    return None


def _is_atomic_expr(s: str) -> bool:
    # Used only for adding/removing parentheses in string conversions.
    return bool(re.fullmatch(r"[-+]?[\w.]+", s))


def _latex_to_ascii(s: str) -> str:
    """
    Convert a small subset of LaTeX math into a more comparable ASCII-ish form.
    This is intentionally conservative (covers common final-answer patterns).
    """
    # \frac{a}{b} -> a/b  (add parentheses when needed)
    i = 0
    while True:
        j = s.find("\\frac", i)
        if j < 0:
            break
        k = j + len("\\frac")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        num, k2 = a
        if k2 >= len(s) or s[k2] != "{":
            i = k2
            continue
        b = _extract_balanced_braces(s, k2)
        if not b:
            i = k2 + 1
            continue
        den, k3 = b
        num_s = _latex_to_ascii(num)
        den_s = _latex_to_ascii(den)
        if not _is_atomic_expr(num_s):
            num_s = f"({num_s})"
        if not _is_atomic_expr(den_s):
            den_s = f"({den_s})"
        rep = f"{num_s}/{den_s}"
        s = s[:j] + rep + s[k3:]
        i = j + len(rep)

    # \sqrt{a} -> sqrt(a)
    i = 0
    while True:
        j = s.find("\\sqrt", i)
        if j < 0:
            break
        k = j + len("\\sqrt")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        inner, k2 = a
        inner_s = _latex_to_ascii(inner)
        rep = f"sqrt({inner_s})"
        s = s[:j] + rep + s[k2:]
        i = j + len(rep)

    return s


def normalize_math_answer(s: str) -> str:
    """
    Best-effort normalizer for MATH-style final answers.

    Goals:
    - Be robust to common LaTeX wrappers (\\boxed, $, \\(\\), \\[\\]).
    - Canonicalize simple numeric/rational answers (e.g., 0.5 == 1/2).
    - Keep non-numeric expressions as a compact, comparable string.
    """
    s = s.strip()
    if not s:
        return s

    # Strip common math wrappers.
    for pat in (_RE_LATEX_INLINE_MATH, _RE_LATEX_PARENS_MATH, _RE_LATEX_BRACKETS_MATH):
        m = pat.match(s)
        if m:
            s = m.group(1).strip()
            break

    # Drop common LaTeX spacing and sizing commands.
    s = re.sub(r"\\(left|right)\b", "", s)
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\ ", " ")

    # Normalize common operators/symbols.
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\pm", "+-")

    # Normalize fraction command variants.
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Remove remaining whitespace early (post replacements), but keep newlines out.
    s = re.sub(r"\s+", "", s)

    # Remove outer braces.
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # Replace LaTeX braces with plain braces (we'll strip later).
    s = s.replace("\\{", "{").replace("\\}", "}")

    s = _latex_to_ascii(s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # After minimal LaTeX conversion, try numeric/rational canonicalization again.
    # Attempt to canonicalize pure numeric / rational strings.
    # Fraction() handles "a/b" and decimals; it will raise on expressions like "(1+2)/3".
    try:
        f = Fraction(s)
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        pass

    # Final cleanup: remove redundant braces.
    s = s.replace("{", "").replace("}", "")
    return s.strip()
