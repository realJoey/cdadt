"""Write the badge data ``README.md`` displays, from what a CI run actually measured.

Why this runs in CI and is never committed
------------------------------------------

These badges used to be SVG files generated here and committed, because ``cdadt`` was a private
repository and shields.io -- an anonymous third party -- could not read it. That is no longer
true, and a committed badge has a defect the privacy argument was paying for: it can only ever
restate a *configured* value. The coverage badge in particular reported ``fail_under``, the
threshold the suite enforces, rather than the coverage a run achieved. Honest, and one step
removed from the thing a reader assumes they are being told.

So the two values that only a run can know are emitted by the run. ``full`` measures coverage
over the whole suite, this script turns that measurement into shields.io *endpoint* documents,
and they are published as part of the documentation site. ``README.md`` points shields.io at
those URLs, so what the badge says is what the last fully-gated commit on ``main`` measured.

Nothing here is committed: the output lands inside ``docs/_build/html``, which is ignored. A
badge that no longer has a run behind it therefore goes stale visibly -- shields.io renders the
fetch failure -- rather than continuing to display a number nobody has re-measured.

The other badges bypass this script because a live source already exists: the licence and the CI
status come from GitHub, the Python floor from ``pyproject.toml`` over raw.githubusercontent,
``ruff``'s from the badge ``ruff`` itself publishes, and the docs badge from whether the site
this script writes into is reachable.

Usage
-----

.. code-block:: bash

   coverage json -o coverage.json
   python docs/badges/publish_badges.py coverage.json docs/_build/html/badges

Rendering is not this script's job, which is the other thing that changed. It writes the four
fields of shields.io's endpoint schema and lets shields.io draw the rectangle.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: The shields.io endpoint schema version these documents are written against. shields.io reads
#: this field and will refuse a document that claims a version it does not implement, so it is a
#: compatibility statement rather than decoration.
SCHEMA_VERSION = 1


def coverage_colour(percent: float) -> str:
    """Return the shields.io colour for a coverage *percent*.

    Banded rather than a single colour, even though ``fail_under = 100`` means a published badge
    is always at the top band: the bands are what make a lowered threshold visible. If that gate
    is ever relaxed, the badge should look different, not merely say a different number.

    Parameters
    ----------
    percent : float
        Measured coverage, 0 to 100.

    Returns
    -------
    str
        A shields.io colour name.
    """
    if percent >= 100:
        return "brightgreen"
    if percent >= 90:
        return "green"
    if percent >= 75:
        return "yellowgreen"
    if percent >= 60:
        return "yellow"
    return "red"


def coverage_badge(coverage_json: Path) -> dict:
    """Return the endpoint document for the coverage badge.

    Parameters
    ----------
    coverage_json : Path
        The report written by ``coverage json``. ``totals.percent_covered`` is used rather than
        ``percent_covered_display``, which is rounded to a whole number by coverage itself: 99.6%
        must not be able to present itself as ``100%`` on the one badge where that distinction is
        the whole point.

    Returns
    -------
    dict
        A shields.io endpoint document.
    """
    totals = json.loads(coverage_json.read_text(encoding="utf-8"))["totals"]
    percent = float(totals["percent_covered"])

    # Truncated, never rounded, and "100%" is reachable only from an exact 100. Formatting with
    # `f"{percent:.1f}"` would render 99.96 as "100.0" -- a badge claiming total coverage of a run
    # that missed a line, which is the single failure this badge exists to make impossible.
    message = "100%" if percent >= 100 else f"{math.floor(percent * 10) / 10:.1f}%"

    return {
        "schemaVersion": SCHEMA_VERSION,
        "label": "coverage",
        "message": message,
        "color": coverage_colour(percent),
    }


def version_badge() -> dict:
    """Return the endpoint document for the version badge.

    The version is *imported* rather than parsed out of a file. ``pyproject.toml`` declares it
    ``dynamic`` and resolves it from ``cdadt.__version__``, so importing is what the installed
    distribution would report -- and a badge that disagreed with the installed package would be
    worse than no badge.

    Returns
    -------
    dict
        A shields.io endpoint document.
    """
    import cdadt

    return {
        "schemaVersion": SCHEMA_VERSION,
        "label": "version",
        "message": cdadt.__version__,
        "color": "blue",
    }


def write(coverage_json: Path, directory: Path) -> list[Path]:
    """Write every endpoint document into *directory*, creating it if needed.

    Parameters
    ----------
    coverage_json : Path
        The report written by ``coverage json``.
    directory : Path
        Where to write. In CI this is inside the built documentation site, so the documents are
        published with it and are reachable at ``<site>/badges/<name>.json``.

    Returns
    -------
    list of Path
        The files written, in the order written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    written = []
    for name, document in (("coverage", coverage_badge(coverage_json)), ("version", version_badge())):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    return written


def main(argv: list[str]) -> int:
    """Write the badge documents named by *argv*.

    Parameters
    ----------
    argv : list of str
        ``[coverage_json, output_directory]``. Both are required rather than defaulted: this runs
        inside a workflow, where a silently wrong default path would publish a stale badge instead
        of failing.

    Returns
    -------
    int
        A process exit status.
    """
    if len(argv) != 2:
        print("usage: publish_badges.py <coverage.json> <output directory>", file=sys.stderr)
        return 2

    coverage_json, directory = Path(argv[0]), Path(argv[1])
    if not coverage_json.is_file():
        print(f"no coverage report at {coverage_json}; run `coverage json -o {coverage_json}` first", file=sys.stderr)
        return 1

    for path in write(coverage_json, directory):
        print(f"wrote {path}")
    return 0


# No `# pragma: no cover` here, unlike the two lines docs/verification.rst names: `source` in
# [tool.coverage.run] is `cdadt`, so nothing under docs/ is measured and an exclusion would be
# claiming to exempt a line from a gate that never looked at it.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
