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
    ap.add_argument("--slot", help="only show one slot (hero/ortho/detail/material)")
    args = ap.parse_args()

    print(f"build {BUILD}   margin {SUBJECT_MARGIN}   floor {SUBJECT_FLOOR}\n")

    validator = ReferenceValidator()
    taxonomy = TaxonomyEngine(validator.model, validator.tokenizer, validator.device)

    queries = await taxonomy.generate_reference_queries(args.prompt)
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
    scored = validator.score_candidates(survivors, args.prompt)

    board = select_board(scored, args.count)
    print(f"\nBoard: {len(board)} images")
    for item in board:
        print(f"  {item.slot:<9} subject {item.subject:.3f}  style {item.score:.3f}  "
              f"{item.url[:66]}")

    print("\nPer-slot subject range among kept images:")
    for slot in sorted({s.slot for s in scored}):
        vals = [s.subject for s in scored if s.slot == slot]
        print(f"  {slot:<9} {min(vals):.3f} .. {max(vals):.3f}   ({len(vals)} kept)")


if __name__ == "__main__":
    asyncio.run(main())