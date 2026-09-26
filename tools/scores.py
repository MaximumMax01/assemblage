"""
Score inspector.

NB: this file must not be named inspect.py. Python puts a script's own
directory first on sys.path, so tools/inspect.py shadows the standard library's
inspect module for everything imported afterwards -- including asyncio, which
imports it internally. The failure surfaces far from the cause, as
"partially initialized module 'inspect' has no attribute 'signature'".

Prints every candidate image for a prompt with its subject similarity, its slot
style score, and whether it was kept or why it was dropped. Use it to pick
threshold values from real numbers instead of guessing.

    python tools/scores.py "office ceiling tiles"
    python tools/scores.py "office ceiling tiles" --slot detail

Reading the output: find a row you know is wrong (the castle roof) and a row you
know is right (the ceiling grid closeup). The subject column for those two is
the range your threshold has to separate. If the bad row's subject score is
close to the good one's, no threshold will split them and the fix belongs in the
query, not the filter.
"""

import argparse
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, ROOT)

from backend.net import apply_ssl_workarounds

apply_ssl_workarounds()

from backend.scraper import fetch_all_candidates
from backend.slots import profile_for
from backend.taxonomy import TaxonomyEngine
from backend.validator import (
    BUILD, SUBJECT_FLOOR, SUBJECT_MARGIN, ReferenceValidator, select_board,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--slot", help="only show one slot id, e.g. ortho or plan_simple")
    args = ap.parse_args()

    print(f"build {BUILD}   margin {SUBJECT_MARGIN}   floor {SUBJECT_FLOOR}\n")

    validator = ReferenceValidator()
    taxonomy = TaxonomyEngine(validator.model, validator.tokenizer, validator.device)

    plan = taxonomy.plan(args.prompt)
    queries = plan.queries
    print(f"\nMode: {plan.mode}   archetype: {plan.archetype} ({plan.kind_label})   subject: {plan.subject!r}")
    for h in plan.hints:
        print(f"  hint ({h['kind']}): {h['text']}")
        if h["suggestions"]:
            print(f"    try: {', '.join(h['suggestions'])}")
    print("\nQueries")
    for q in queries:
        print(f"  {profile_for(q.slot)['label']:<26} {q.query}")

    candidates = await fetch_all_candidates(queries)
    survivors = validator.prefilter_all(candidates)
    if args.slot:
        survivors = [s for s in survivors if s[2] == args.slot]

    if not survivors:
        print("\nNothing survived the geometry and sharpness gates.")
        return

    # Force the per-image table on. Import the same module object the
    # validator instance came from -- backend/ is also on sys.path here, so
    # `validator` and `backend.validator` would otherwise be two separate
    # modules with independent globals, and setting VERBOSE on the wrong one
    # would silently do nothing.
    import backend.validator as V
    assert V.ReferenceValidator is ReferenceValidator, "duplicate module import"
    V.VERBOSE = True
    print()
    scored = validator.score_candidates(survivors, plan.subject, plan.mode,
                                        plan.subject_templates, plan.extra_negatives)

    board = select_board(scored, args.count, [q.slot for q in queries])
    print(f"\nBoard: {len(board)} images")
    for item in board:
        tag = "  [reserve]" if item.gated else ""
        print(f"  {item.slot:<9} subject {item.subject:.3f}  style {item.score:.3f}  "
              f"combined {item.combined:.3f}  {item.url[:52]}{tag}")

    print("\nPer-slot subject range:")
    for slot in sorted({s.slot for s in scored}):
        clean = [s.subject for s in scored if s.slot == slot and not s.gated]
        reserve = [s.subject for s in scored if s.slot == slot and s.gated]
        line = f"  {slot:<9} clean {len(clean):>2}"
        if clean:
            line += f" ({min(clean):.3f}..{max(clean):.3f})"
        line += f"   reserve {len(reserve):>2}"
        if reserve:
            line += f" ({min(reserve):.3f}..{max(reserve):.3f})"
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
