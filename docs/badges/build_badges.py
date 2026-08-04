"""Generate the status badges ``README.md`` displays.

Why these are files in the repository rather than ``shields.io`` URLs
---------------------------------------------------------------------

The usual way to put a badge on a README is to point an ``<img>`` at ``shields.io``, which reads
the repository and renders the answer live. This repository is public, so that route is open to
it -- these badges are files by choice, not by necessity. The original reason was that an
anonymous third party could not read a private repository at all; the reason they stayed is
different, and better.

What these four badges report -- the Python floor, the licence, the version, the coverage
threshold -- are all facts *this repository defines*. A shields.io badge would restate them from
somewhere else, on someone else's schedule, with no way for the suite to notice when the two
disagreed. Deriving and committing them keeps the statement and its source in the same commit.

The ``tests`` badge is the exception, and it is not built here. GitHub serves it from the real
outcome of ``.github/workflows/ci.yml``. A test result is the only status that changes without
any file changing, so it is the only one that has to be live.

The rest are generated here and committed. A committed badge is a *claim*, and a claim that
nothing checks is exactly the kind of decoration this project has been bitten by before. Two
things stop that:

* Every value is **derived**, never typed. The Python floor comes from ``requires-python``, the
  licence from ``LICENSE``, the version from ``cdadt.__version__``, and the coverage figure from
  ``fail_under`` in ``[tool.coverage.report]``. Nothing here is a literal that could drift.
* ``tests/test_docs.py`` re-derives all four from the same sources and fails if a committed badge
  disagrees. Regenerating them is therefore not optional maintenance; the suite enforces it.

The coverage badge deserves its own note, because it is the one that would be easiest to fake.
It does not report a measured number. It reports ``fail_under``, which the suite is configured to
*enforce* -- coverage below it makes ``coverage report`` exit non-zero. So "100%" is not a
recollection of a good run, it is a threshold the gate would refuse to pass beneath, and the test
below asserts the badge matches that threshold rather than matching a number someone once saw.

Usage
-----

.. code-block:: bash

   python docs/badges/build_badges.py

Rendering is deliberately dependency-free: the SVG is written directly rather than through
``anybadge`` or ``genbadge``, because a badge is a rectangle and two pieces of text, and adding a
package to the environment to draw one is not a trade worth making.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

#: Shields' own palette, so these sit beside conventional badges without looking foreign.
BRIGHT_GREEN = "#4c1"
BLUE = "#007ec6"
BLACK = "#000000"
RUFF_PURPLE = "#261230"


class TextMeasure:
    """Approximate the rendered width of a string in 11px Verdana.

    A badge is only the right shape if the coloured boxes fit their text, and the text is
    centred. Doing that exactly needs font metrics; doing it well enough needs a table of
    advance widths, which is what this is. The characters badges actually contain -- digits,
    ASCII letters, ``.``, ``+``, ``%`` -- are given measured-ish values and everything else
    falls back to the average.
    """

    #: Advance widths in pixels at 11px, for the characters that appear in these badges.
    WIDTHS: ClassVar[dict[str, float]] = {
        " ": 5.0,
        ".": 3.5,
        "+": 7.3,
        "%": 11.0,
        "-": 4.3,
        "i": 3.2,
        "j": 3.2,
        "l": 3.2,
        "f": 4.0,
        "r": 4.6,
        "t": 4.3,
        "m": 10.5,
        "w": 8.5,
        "M": 10.5,
        "W": 11.0,
        "I": 4.0,
    }

    #: Fallback widths for characters not named above.
    DIGIT = 7.0
    UPPER = 8.0
    LOWER = 6.5

    def width(self, text: str) -> float:
        """Return the approximate rendered width of ``text`` in pixels."""
        total = 0.0
        for character in text:
            if character in self.WIDTHS:
                total += self.WIDTHS[character]
            elif character.isdigit():
                total += self.DIGIT
            elif character.isupper():
                total += self.UPPER
            else:
                total += self.LOWER
        return total


class Badge:
    """One badge: a grey label box, a coloured value box, and the SVG that draws them."""

    #: Horizontal padding inside each box, in pixels.
    PADDING = 10.0

    def __init__(self, label: str, value: str, colour: str, measure: TextMeasure | None = None) -> None:
        """Store what the badge says and how it is coloured.

        Parameters
        ----------
        label : str
            The left, grey half -- what is being reported.
        value : str
            The right, coloured half -- the status itself.
        colour : str
            CSS colour for the value box.
        measure : TextMeasure, optional
            Injected so the geometry can be exercised independently of the drawing.
        """
        self._label = label
        self._value = value
        self._colour = colour
        self._measure = measure or TextMeasure()

    @property
    def filename(self) -> str:
        """Return the file this badge is written to, derived from its label."""
        return re.sub(r"[^a-z0-9]+", "_", self._label.lower()).strip("_") + ".svg"

    def svg(self) -> str:
        """Return the badge as a self-contained SVG document.

        Text is drawn twice -- once in near-black at one-third opacity a pixel lower, then in
        white -- which is the drop shadow every flat badge uses to stay legible against both
        halves.
        """
        label_width = round(self._measure.width(self._label) + 2 * self.PADDING)
        value_width = round(self._measure.width(self._value) + 2 * self.PADDING)
        total = label_width + value_width
        label_centre = label_width / 2
        value_centre = label_width + value_width / 2
        described = f"{self._label}: {self._value}"

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" '
            f'role="img" aria-label="{described}">'
            f"<title>{described}</title>"
            '<linearGradient id="s" x2="0" y2="100%">'
            '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
            '<stop offset="1" stop-opacity=".1"/>'
            "</linearGradient>"
            f'<clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>'
            '<g clip-path="url(#r)">'
            f'<rect width="{label_width}" height="20" fill="#555"/>'
            f'<rect x="{label_width}" width="{value_width}" height="20" fill="{self._colour}"/>'
            f'<rect width="{total}" height="20" fill="url(#s)"/>'
            "</g>"
            '<g fill="#fff" text-anchor="middle" '
            'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
            f'<text x="{label_centre}" y="15" fill="#010101" fill-opacity=".3">{self._label}</text>'
            f'<text x="{label_centre}" y="14">{self._label}</text>'
            f'<text x="{value_centre}" y="15" fill="#010101" fill-opacity=".3">{self._value}</text>'
            f'<text x="{value_centre}" y="14">{self._value}</text>'
            "</g>"
            "</svg>\n"
        )


class RepositoryFacts:
    """The badge values, read from the files that define them rather than written down here.

    Every property is a single statement of where a fact lives. That is the whole point: if the
    licence changes, or the Python floor moves, or the coverage gate is loosened, the badge
    follows without anyone remembering it exists -- and the test that mirrors these properties
    fails until it is regenerated.
    """

    def __init__(self, root: Path = ROOT) -> None:
        """Read the project metadata once.

        Parameters
        ----------
        root : Path
            Repository root, injected so tests can point this at a fixture tree.
        """
        self._root = root
        with open(root / "pyproject.toml", "rb") as handle:
            self._pyproject = tomllib.load(handle)

    @property
    def version(self) -> str:
        """Return the package version, read from ``cdadt/__init__.py``.

        By regex rather than by import, so the badges can be built in an interpreter that has
        none of cdadt's dependencies installed. ``pyproject.toml`` declares the version
        ``dynamic`` and points at this same attribute, so there is still only one source.
        """
        text = (self._root / "cdadt" / "__init__.py").read_text(encoding="utf-8")
        found = re.search(r'^__version__ = "([^"]+)"', text, re.M)
        if found is None:
            raise ValueError("cdadt/__init__.py does not define __version__")
        return found.group(1)

    @property
    def python_requirement(self) -> str:
        """Return the supported Python range, as ``requires-python`` states it."""
        return self._pyproject["project"]["requires-python"].lstrip(">=") + "+"

    @property
    def licence(self) -> str:
        """Return the licence name, from the first line of the file ``pyproject.toml`` points at."""
        named = self._pyproject["project"]["license"]["file"]
        first_line = (self._root / named).read_text(encoding="utf-8").splitlines()[0].strip()
        return first_line.removesuffix(" License")

    @property
    def coverage_threshold(self) -> str:
        """Return the coverage the suite refuses to pass beneath, from ``fail_under``."""
        return f"{self._pyproject['tool']['coverage']['report']['fail_under']}%"


class BadgeSet:
    """The badges ``README.md`` shows, and the act of writing them to disk."""

    def __init__(self, facts: RepositoryFacts | None = None, directory: Path = HERE) -> None:
        """Assemble the set.

        Parameters
        ----------
        facts : RepositoryFacts, optional
            Where the values come from. Injected so the set can be built against a fixture.
        directory : Path
            Where the SVGs are written.
        """
        self._facts = facts or RepositoryFacts()
        self._directory = directory

    def badges(self) -> list[Badge]:
        """Return every badge, in the order the README displays them.

        There is deliberately no ``tests`` badge here. That one is live -- GitHub Actions renders
        it from the actual outcome of ``.github/workflows/ci.yml`` -- because a test result is
        the one status that genuinely changes without any file in the repository changing, and a
        stored copy of it would be a claim about a run nobody can point to.

        Everything below is different in kind. Each is a *declaration* the repository makes about
        itself, so it changes only when a tracked file changes, and regenerating alongside that
        file is exactly right.
        """
        return [
            Badge("coverage", self._facts.coverage_threshold, BRIGHT_GREEN),
            Badge("python", self._facts.python_requirement, BLUE),
            Badge("license", self._facts.licence, BLUE),
            Badge("code style", "black", BLACK),
            Badge("linting", "ruff", RUFF_PURPLE),
            Badge("docs", "sphinx", BLUE),
            Badge("version", self._facts.version, BLUE),
        ]

    def write(self) -> list[Path]:
        """Write every badge and return the paths written."""
        self._directory.mkdir(parents=True, exist_ok=True)
        written = []
        for badge in self.badges():
            path = self._directory / badge.filename
            path.write_text(badge.svg(), encoding="utf-8")
            written.append(path)
        return written


def main() -> int:
    """Write the badge set and report what was written."""
    for path in BadgeSet().write():
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
