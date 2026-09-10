"""Fail if prose is crowding out the code.

Documentation is not the problem; documentation living in the wrong place is.
Four kinds of prose, four homes:

    docstrings          what this callable is and what it takes. One summary line
                        plus Args/Returns where the types do not already say it.
    docs/adr/           *why* a design choice was made.
    docs/explainers/    what a method is and how to read its output -- the physics
                        and the statistics.
    docs/notes/         working notes and open questions. Not published.

This enforces the first of those. When a docstring outgrows the cap, the surplus
is almost always rationale or a tutorial, and belongs in one of the other three.

    uv run python scripts/check_docs.py
    uv run python scripts/check_docs.py --report   # ranked, no failure

Static: parses with ast, imports nothing.
"""

from __future__ import annotations

import argparse
import ast
import io
import sys
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qvarnet"

MAX_DOCSTRING_LINES = 20  # a summary plus an honest Args block for ~8 params
# The per-docstring cap is the primary gate: it is what tracks navigability, since
# what makes a file hard to read is one 60-line essay, not ten honest Args blocks.
# The share cap is only a backstop against a file that is mostly narrative -- a module
# of many small, well-documented functions legitimately runs near 50%.
MAX_PROSE_SHARE = 0.50

# A module docstring is the one place a short orientation paragraph earns its keep,
# so it gets more room than a function's -- but not unlimited room.
MAX_MODULE_DOCSTRING_LINES = 24


def measure(path: Path):
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comments = sum(
        1
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT
    )

    doc_lines = 0
    offenders = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        text = ast.get_docstring(node, clean=False)
        if not text:
            continue
        n = text.count("\n") + 2
        doc_lines += n
        is_module = isinstance(node, ast.Module)
        cap = MAX_MODULE_DOCSTRING_LINES if is_module else MAX_DOCSTRING_LINES
        if is_module and path.name == "__init__.py" and path.parent == SRC:
            # The package front page is the one docstring a reader arrives at
            # cold; an orientation paragraph and a worked example earn their keep.
            continue
        if n > cap:
            offenders.append(
                (getattr(node, "name", "<module>"), getattr(node, "lineno", 1), n, cap)
            )

    share = (doc_lines + comments) / total if total else 0.0
    return {
        "total": total,
        "doc": doc_lines,
        "comments": comments,
        "code": total - blank - comments - doc_lines,
        "share": share,
        "offenders": offenders,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="rank files; never fail")
    args = ap.parse_args()

    results = {p: measure(p) for p in sorted(SRC.rglob("*.py"))}
    total = sum(r["total"] for r in results.values())
    doc = sum(r["doc"] for r in results.values())
    comments = sum(r["comments"] for r in results.values())
    code = sum(r["code"] for r in results.values())

    print(
        f"{total} lines: {code} code, {doc} docstring, {comments} comment "
        f"-> prose share {100 * (doc + comments) / total:.1f}%"
    )

    if args.report:
        print(f"\n{'prose':>6}{'doc':>6}{'worst':>7}  file")
        ranked = sorted(results.items(), key=lambda kv: -kv[1]["share"])
        for path, r in ranked[:20]:
            worst = max((o[2] for o in r["offenders"]), default=0)
            print(f"{100 * r['share']:>5.0f}%{r['doc']:>6}{worst:>7}  {path.relative_to(SRC)}")
        return 0

    failures = []
    for path, r in results.items():
        rel = path.relative_to(SRC)
        for name, lineno, n, cap in r["offenders"]:
            failures.append(f"  {rel}:{lineno}  {name}() docstring is {n} lines (cap {cap})")
        # Only meaningful for files with real code in them: a 20-line abstract
        # base class is legitimately mostly docstring, and that is fine.
        if r["share"] > MAX_PROSE_SHARE and r["code"] >= 60 and not r["offenders"]:
            failures.append(
                f"  {rel}  is {100 * r['share']:.0f}% prose (cap {100 * MAX_PROSE_SHARE:.0f}%)"
            )

    if failures:
        print(f"\n{len(failures)} doc-budget violation(s):\n")
        print("\n".join(sorted(failures)))
        print(
            "\nMove the surplus: rationale -> docs/adr/, physics and how-to-read-it\n"
            "-> docs/explainers/, anything unsettled -> docs/notes/."
        )
        return 1

    print(f"OK: {len(results)} modules within the doc budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
