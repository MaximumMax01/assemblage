<<<<<<< HEAD
# assemblage
=======
# Assemblage

Turns a text prompt into a PureRef board.

Search for reference and you get twenty versions of the same photograph. What you
actually need to model something is four different *kinds* of image: the overall
form, an orthographic or cut-through view, a close-up of how it's joined
together, and a macro of the surface material.

Assemblage searches for all four separately, filters each one on its own terms,
and writes a `.pur` file with the results already laid out and grouped.

```
python app.py
```

Then open <http://localhost:8000>.

---

## Why four searches instead of one

Each prompt is classified into an archetype (architectural, mechanical, organic,
prop) and expanded into four queries, one per slot:

| Slot | What it looks for |
|---|---|
| Form & silhouette | 3/4 view, whole object, neutral lighting |
| Orthographic & technical | blueprints, elevations, cross-sections, plates |
| Joinery & construction | corners, seams, linkages, assembly detail |
| Material & surface | macro texture, wear, photogrammetry scans |

The slot tag stays attached to every image through the whole pipeline, which
matters more than it sounds. Filters tuned for photographs destroy line
drawings: a blueprint is mostly white paper, so a "reject blown-out white
backgrounds" rule deletes it, and it has low edge variance, so a sharpness
threshold set for photos deletes it too. Each slot therefore carries its own
gates and its own CLIP anchors.

The final board is filled by quota rather than by taking the top-scoring images
overall. A photograph of the subject will always outscore a blueprint on a
photographic anchor, so a global ranking returns twelve hero shots and nothing
else — which is the exact problem this tool exists to solve. Slots that can't
fill their quota release their places to the others, so a subject with no
available blueprints still returns a full board.

## Install

Requires Python 3.10+ and [PureRef](https://www.pureref.com/).

```bash
git clone <this repo>
cd assemblage
pip install -r requirements.txt
python app.py
```

First run downloads the CLIP weights (~350 MB) to `~/.cache/huggingface/`.
Everything runs locally. There are no API keys and nothing is uploaded.

### On a network that inspects TLS

School and corporate wifi often break certificate validation. Assemblage
installs the system trust store, which fixes most cases. If it doesn't:

```bash
ASSEMBLAGE_INSECURE_SSL=1 python app.py
```

This disables certificate verification for the process. It's off by default and
should stay that way unless you need it.

## Tests

```bash
python tests/test_pipeline.py
```

`torch` is stubbed, so the tests run without the model installed. They cover the
per-slot gates, duplicate detection, and quota allocation.

## Known limits

- **Search backend fragility.** DuckDuckGo access goes through an unofficial
  wrapper that breaks when their endpoints change. Failures are non-fatal —
  Wikimedia Commons and Openverse carry the load — but results thin out.
- **Datacenter IPs get blocked.** Works fine from a home connection. Hosted on
  a cloud VM, DuckDuckGo will rate-limit or block.
- **Stock watermarks.** CLIP negatives catch most watermarked junk, but stock
  photography with a thin attribution strip along the bottom still gets through.
- **Organic subjects have no blueprints.** The ortho slot looks for scientific
  illustration plates instead, which works less reliably.
- **Thresholds are hand-tuned, not measured.** They come from eyeballing output
  on a handful of prompts. A labelled evaluation set is the obvious next step.

## Licence

MIT. See `LICENSE`.

The `.pur` format support comes from
[FyorDev/PureRef-format](https://github.com/FyorDev/PureRef-format), also MIT.
See `NOTICE` — this project depends entirely on that reverse-engineering work.
>>>>>>> b3ed0a9 (second commit, for school)
