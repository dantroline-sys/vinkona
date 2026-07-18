"""Compatibility gate for the SYSTEM-python surface.

vinkona.sh runs the supervisor with the system python3, and the supervisor
launches the LM services through serve_*.sh whose bare `python3` is ALSO the
system interpreter — on macOS that is 3.9.  Every file on that path (and its
import closure: llm_server loads config.py in-process) must never grow syntax
or imports needing more.  Two Mac mini failures shipped before this gate:
bare `X | Y` annotations evaluate eagerly before 3.10 and crash at import.
Structural checks per file: future-import first, no match statements, no
runtime PEP-604 unions, stdlib-only imports (these files run before/without
any env)."""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["supervisor.py", "llm_server.py", "config.py"]
FAILED = []


def check(label, cond):
    print(("  ok  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def gate(fname):
    tree = ast.parse((HERE / fname).read_text())
    body = tree.body
    first = body[1] if isinstance(body[0], ast.Expr) else body[0]
    check(f"{fname}: `from __future__ import annotations` before other imports",
          isinstance(first, ast.ImportFrom) and first.module == "__future__"
          and any(a.name == "annotations" for a in first.names))

    check(f"{fname}: no match statements (3.10+)",
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
    check(f"{fname}: no runtime type unions outside annotations",
          runtime_unions == [])

    stdlib = getattr(sys, "stdlib_module_names", None)
    roots = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            roots.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            roots.add(n.module.split(".")[0])
    foreign = sorted(r for r in roots
                     if r != "__future__" and stdlib and r not in stdlib)
    check(f"{fname}: imports stdlib only (runs before/without any env)",
          not stdlib or foreign == [])
    if foreign:
        print("   foreign imports:", ", ".join(foreign))


def main():
    for f in FILES:
        gate(f)
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
