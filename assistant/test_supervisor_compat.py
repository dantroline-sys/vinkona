"""Bootstrap-compatibility gate for supervisor.py.

vinkona.sh runs the supervisor with the SYSTEM python3 — on macOS that is
3.9 — so this file must never grow syntax or imports that need more.  The
2026-07 Mac mini failure mode: a bare `int | None` annotation evaluates
eagerly before 3.10 and crashes at import.  This gate keeps it structural:
future-import present, no match statements, no runtime PEP-604 unions, and
stdlib-only imports (the supervisor runs before any env exists)."""
import ast
import sys
from pathlib import Path

SUP = Path(__file__).resolve().parent / "supervisor.py"
FAILED = []


def check(label, cond):
    print(("  ok  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def main():
    tree = ast.parse(SUP.read_text())
    body = tree.body
    first = body[1] if isinstance(body[0], ast.Expr) else body[0]
    check("`from __future__ import annotations` is the first statement",
          isinstance(first, ast.ImportFrom) and first.module == "__future__"
          and any(a.name == "annotations" for a in first.names))

    check("no match statements (3.10+)",
          not any(isinstance(n, ast.Match) for n in ast.walk(tree)))

    # Runtime PEP-604 unions (isinstance(x, A | B), aliases) crash on 3.9 even
    # WITH the future import — only annotations are made lazy.  Collect every
    # BitOr that is not inside an annotation.
    ann_spans = set()
    for n in ast.walk(tree):
        for a in ([getattr(n, "annotation", None), getattr(n, "returns", None)]):
            if a is not None:
                ann_spans.update(id(x) for x in ast.walk(a))
    runtime_unions = [n.lineno for n in ast.walk(tree)
                      if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)
                      and id(n) not in ann_spans
                      and not isinstance(n.left, (ast.Constant, ast.Num))
                      and any(isinstance(x, ast.Name) and x.id in
                              ("int", "str", "float", "dict", "list", "None",
                               "Path", "bool") for x in ast.walk(n))]
    check("no runtime type unions outside annotations", runtime_unions == [])

    stdlib = getattr(sys, "stdlib_module_names", None)
    roots = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            roots.add(n.module.split(".")[0])
    foreign = sorted(r for r in roots
                     if r != "__future__" and stdlib and r not in stdlib)
    check("supervisor imports stdlib only (runs before any env exists)",
          not stdlib or foreign == [])
    if foreign:
        print("   foreign imports:", ", ".join(foreign))

    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
