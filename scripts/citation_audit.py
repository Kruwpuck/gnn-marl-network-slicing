"""
Guard the `path:line` citations that the gate documents use as evidence.

results/GATE_C.md cites source locations by line number to back each verdict --
"agents are constructed with no keyword overrides (training/train_baselines.py:84)".
Line numbers drift every time the cited file is edited, and a drifted pointer looks
exactly like a valid one. On 2026-08-23 six of the C2/C5 pointers had silently moved
1-5 lines; every underlying fact still held, but nothing in the repo would have caught
it if one had not.

Checking that the cited line merely *exists* is too weak (81 -> 84 stays in range), so
this stores an anchor -- the text of each cited line -- in scripts/citation_anchors.json
and fails when the text at that line no longer matches.

  python scripts/citation_audit.py            # check, exit 1 on drift
  python scripts/citation_audit.py --update   # re-anchor after verifying by hand

--update trusts whatever is at the cited line right now. Only run it after re-reading
each moved citation and confirming the claim it supports still holds; a blind refresh
turns this from a guard into a rubber stamp.

Historical documents are skipped (SKIP below): their line numbers were correct when
written, and re-pointing them at today's code would misrepresent what they recorded.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHORS = Path(__file__).with_name("citation_anchors.json")

PKG = "scripts|training|envs|agents|gnn|evaluation|traffic|tests|configs|ablation"
CITE = re.compile(rf"\b(?P<path>(?:{PKG})/[\w/]+\.(?:py|yaml)):(?P<lines>\d+(?:[,-]\d+)*)")

# Snapshots, not live claims: correct when written, wrong to renumber. See docs/INDEX.md.
SKIP = (
    "docs/rev2-implementation-plan.md",
    "docs/archive/",
    "docs/journey/",
    "results/v1_uncoupled/",
    "results/v2_scalarized/",
    "results/quarantine_eps1.0/",
    "runs/",  # append-only ledger
)


def tracked_markdown() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [d for d in out.splitlines() if not d.startswith(SKIP)]


def collect() -> dict[str, dict[str, str | None]]:
    """Map doc -> {"path:line": text at that line}. Missing/out-of-range become None."""
    found: dict[str, dict[str, str | None]] = {}
    source: dict[str, list[str] | None] = {}
    for doc in tracked_markdown():
        for line in (ROOT / doc).read_text(encoding="utf-8", errors="replace").splitlines():
            for m in CITE.finditer(line):
                path = m.group("path")
                if path not in source:
                    p = ROOT / path
                    text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
                    source[path] = text.splitlines() if text is not None else None
                for n in re.split(r"[,-]", m.group("lines")):
                    i = int(n)
                    src = source[path]
                    at = src[i - 1].strip() if src and 0 < i <= len(src) else None
                    found.setdefault(doc, {})[f"{path}:{i}"] = at
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-anchor to the current lines")
    args = ap.parse_args()

    found = collect()

    if args.update:
        ANCHORS.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        n = sum(len(v) for v in found.values())
        print(f"anchored {n} citations across {len(found)} documents -> {ANCHORS.name}")
        return 0

    if not ANCHORS.exists():
        print(f"no anchor file; run: python scripts/{Path(__file__).name} --update", file=sys.stderr)
        return 1

    expected = json.loads(ANCHORS.read_text(encoding="utf-8"))
    problems: list[str] = []

    for doc, cites in found.items():
        for cite, actual in cites.items():
            want = expected.get(doc, {}).get(cite, "<not anchored>")
            if actual is None:
                problems.append(f"{doc}: {cite} does not exist (file missing or line out of range)")
            elif want == "<not anchored>":
                problems.append(f"{doc}: {cite} is new and unanchored -> {actual!r}")
            elif actual != want:
                problems.append(f"{doc}: {cite} drifted\n    was {want!r}\n    now {actual!r}")

    for doc, cites in expected.items():
        for cite in cites:
            if cite not in found.get(doc, {}):
                problems.append(f"{doc}: {cite} anchored but no longer cited (stale anchor)")

    if problems:
        print(f"CITATION DRIFT ({len(problems)}):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nRe-read each claim at its new line. If it still holds, re-anchor with --update.",
              file=sys.stderr)
        return 1

    total = sum(len(v) for v in found.values())
    print(f"OK: {total} citations across {len(found)} documents match their anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
