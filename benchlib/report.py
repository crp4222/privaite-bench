"""Console/markdown formatting shared by the runners.

`pct` is the safe-divide that appeared ~15 times (`num/den*100 if den else 0.0`);
`rule` is the `"=" * width` banner line. Both return exactly what the inline code
returned, so stdout and generated markdown stay byte-for-byte identical.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def pct(num: float, den: float, ndigits: int | None = None) -> float:
    """`num/den*100`, or 0.0 when `den` is 0. Rounds to `ndigits` if given, else
    returns the raw float (callers that never rounded pass no ndigits)."""
    if not den:
        return 0.0
    value = num / den * 100
    return round(value, ndigits) if ndigits is not None else value


def rule(width: int, char: str = "=") -> str:
    """A banner line, e.g. `rule(72)` -> 72 '=' characters."""
    return char * width


def md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Standard GitHub markdown table as a list of lines: header, `|---|` separator,
    then one line per row. Cells are used verbatim (callers pre-format numbers)."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines
