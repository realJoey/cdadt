Developing cdadt
================

This page is the working guide: what to run, what has to pass, and where a change of each kind
belongs. :doc:`architecture` says why the design is the way it is; this says how to work inside
it.

It assumes you have an environment already. If not, :doc:`install` builds one, and the OpenBLAS
pin there is not optional on Windows.

.. contents::
   :local:
   :depth: 1

The loop
--------

.. code-block:: bash

   conda activate cdadt_env

   pytest -q -m "not slow"      # the fast loop: 320 tests, about three minutes
   pytest -q                    # everything: 384 tests, about twenty-five minutes
   pytest -q --cov=cdadt        # everything, with the coverage gate
   ruff check cdadt tests       # lint
   black cdadt tests            # format
   cd docs && make html         # or: sphinx-build -W -b html docs docs/_build/html

Work against ``-m "not slow"`` while you are changing something, and run the full suite with
coverage before you commit. The slow tests are the ones that build and converge a real
OpenConcept model; they are the ones that catch a claim about the real mission being wrong, so
they are not optional before a commit, only during one.

What a change has to pass
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Gate
     - What it means
   * - ``pytest -q``
     - All 384 pass. A test that is slow is marked ``slow``, not deleted.
   * - ``pytest --cov=cdadt``
     - **100%** of statements and branches. ``fail_under = 100`` is in ``pyproject.toml``, so
       this fails the run rather than reporting a number. See :doc:`verification` for the two
       lines that carry ``# pragma: no cover`` and why.
   * - ``ruff check``
     - Clean. No ``# noqa`` without a reason beside it.
   * - ``black``
     - Clean. Line length is in ``pyproject.toml``; do not fight it by hand.
   * - ``sphinx-build -W``
     - Docs build with warnings as errors, so a broken cross-reference fails rather than
       rotting.

The coverage gate is the one that most often has something to say. It is defensible here
*because* cdadt computes no physics: there is no solver to drive into an exotic state, only
interface, validation, routing and reporting, all of which are reachable from a test. A line you
cannot reach is either dead code or a missing test.

A trap worth knowing: a test written to cover a specific line can pass while that line stays
uncovered, because an earlier guard rejected the input first. When you are targeting a named
line, the acceptance criterion is the coverage report, not the green test:

.. code-block:: bash

   pytest -q --cov=cdadt.config --cov-report=term-missing

Where a change belongs
----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - You want to
     - Do this
   * - Change an aeroplane, a mission, or what is optimized
     - Edit a case file. No Python. See :doc:`configuration` and :doc:`tutorials`.
   * - Add a certification requirement
     - A ``constraints`` entry, with ``regulation`` and ``source`` if it is to count as
       evidence. See :doc:`certification`.
   * - Add an engineering domain
     - Subclass :class:`~cdadt.disciplines.base.Discipline`, declare ``discipline_name``,
       ``owned_patterns`` and ``reported``, and pass it to
       :class:`~cdadt.aircraft.Aircraft`. Nothing existing changes.
   * - Add a command
     - Subclass :class:`~cdadt.cli.Command` (or :class:`~cdadt.cli.StudyCommand`) and add it to
       :data:`~cdadt.cli.DEFAULT_COMMANDS`. Nothing existing changes.
   * - Drive a different analysis
     - Change ``black_box.model`` in the case file. cdadt imports no OpenConcept module, so
       there is nothing to subclass or re-wire.
   * - Add an output file
     - A method on :class:`~cdadt.artifacts.StudyArtifacts`, called from
       :meth:`~cdadt.cli.StudyCommand.archive`. See :doc:`artifacts`.
   * - Supply your own aerodynamics
     - Subclass :class:`~cdadt.models.loads.AerodynamicLoads` in :mod:`cdadt.models` and name it
       in the case file. Analytic derivatives are required. See :doc:`aerodynamics`.

Three of those are open/closed on purpose -- a new discipline, a new command and a new black box
each require editing nothing that already exists -- and each has a test that adds one to prove
it. If you find yourself editing an existing class to add a new thing of the same kind, that is
worth stopping over.

Rules that are not style preferences
-------------------------------------

**Never modify anything under** ``openconcept/``. It is an immutable third-party dependency. A
contract test checks the working tree is clean and that any commit it carries beyond upstream
does not touch a module cdadt loads. If OpenConcept has a bug, the fix belongs in a cdadt
wrapper, not in the clone.

**Do not copy it, and do not patch it either.** Both are ways of getting a dependency's
behaviour without importing, subclassing or editing it, so both would slip past the checks
above -- a copy has no import to find and leaves the clone spotless; a patch leaves the source
of both untouched. Each now has its own contract test, and the second matters most for
:doc:`validation`: a patched dependency means the reference run and the cdadt run are no longer
executing the same code, which is the one assumption that whole comparison rests on.

**No cdadt module imports OpenConcept.** The analysis is named in the case file as
``module:ClassName`` and loaded at run time. A contract test parses every source file to enforce
this. It is what makes "black box" a fact rather than a claim.

**Physics stays in the box.** A discipline has no ``compute``, no residual and no partial
derivative. It is the validated interface to its slice of the box. A ``cdadt.disciplines``
module that looked like it modelled aerodynamics, in a tool whose numbers get quoted in a
thesis, would be one page of documentation away from a false claim.

**Documentation is part of the change, not after it.** A new public class belongs in
:doc:`architecture` and in an API page under ``docs/api/``, in the same commit. The
architecture page has been wrong before precisely because a module was added without it.

**Numbers in the documentation are measured.** Every figure in :doc:`verification` and
:doc:`optimization` comes from a run, not from a previous version of the page. If you change
something that moves a number, re-run and re-read it; do not scale the old one.

Writing a test
--------------

Every test carries exactly one marker, and the marker is a claim about what it establishes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - The claim
   * - ``unit``
     - One cdadt class behaves as specified, with no model built.
   * - ``contract``
     - The cdadt/OpenConcept boundary and the ownership map hold.
   * - ``integration``
     - A real model builds, converges and is driven.
   * - ``verification``
     - The equations are solved right.
   * - ``validation``
     - The right equations were solved.

Add ``slow`` as well when a test builds and converges the real model, so the fast loop stays
fast.

**The suite runs from a temporary working directory.** A session-scoped fixture in
``conftest.py`` moves it there, because a driver writes its own log -- pyOptSparse puts
``IPOPT.out`` into the problem's output directory -- and OpenMDAO creates that directory on
demand, named after the *problem*, wherever the process happens to be. ``reports=False`` does
not prevent it; it suppresses the reports, not the directory a driver writes into. Every
optimization test used to leave a ``__main__<n>_out`` folder in the repository root. Everything
the suite reads is addressed absolutely, so nothing depends on where it runs from.

Prefer a real object to a double. Where these tests use a fake -- ``FakeBox`` in
``test_artifacts.py`` and ``test_certification.py`` -- it is to keep the test about cdadt's own
logic, and each is mirrored by an integration test that makes the same claim against the real
box. A claim about the real mission cannot be made against a fake, and a claim about what
OpenMDAO accepts cannot be made against a recorder that only proves a method was called.

Do not reach into a dependency's private attributes from a test. If the only way to see
something is through an underscore, build the thing properly and read it through the public API
instead -- it is slower and it is correct.

Committing
----------

One logical change per commit, files staged by name, pushed immediately:

.. code-block:: bash

   git add path/to/file            # never -A, never .
   git commit
   git push

Write the message in the imperative, subject under 72 characters, and say *why* rather than
what -- the diff already says what. The commit messages in this repository's history are the
worked examples.

``__pycache__``, coverage output and built docs are all in ``.gitignore``. ``run_outputs/`` is
**not**: the shipped runs are committed, so the results the documentation quotes can be read
without re-running anything. That costs roughly 4 MB per run and git never forgets, so add runs
deliberately rather than by habit.

Two ``.gitignore`` entries are anchored with a leading slash -- ``/reports/`` and ``/*_out/`` --
and must stay that way. A pattern without one matches at *every* depth, and these two previously
swallowed ``run_outputs/<case>_<stamp>_out/`` and the ``reports/`` inside each. Anchored, they
still catch a run started from the wrong directory, which is what they were for.

Generated artefacts
-------------------

One directory holds files that are committed but never hand-edited. They regenerate from one
command, and the suite fails if the committed copy has drifted:

.. code-block:: bash

   python docs/xdsm/build_xdsm.py      # the three XDSM figures

The badges
----------

Every badge in ``README.md`` is live, and nothing about them is committed. Each is a shields.io
URL pointing at whatever source actually defines the value:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Badge
     - Source
     - Why there
   * - ``CI``
     - GitHub, from ``ci.yml``'s real outcome
     - A test result is the only status that changes without any file changing.
   * - ``coverage``
     - ``<site>/badges/coverage.json``
     - Only ``full`` runs the whole suite under coverage, so only ``full`` can measure it.
   * - ``version``
     - ``<site>/badges/version.json``
     - ``pyproject.toml`` declares the version ``dynamic``; it resolves from ``cdadt.__version__``.
   * - ``python``
     - ``pyproject.toml`` over raw.githubusercontent
     - ``requires-python`` is the declaration; the badge reads it directly.
   * - ``license``
     - GitHub's own licence detection
     - Same reason.
   * - ``linting``
     - The document ``ruff`` publishes for the purpose
     - Upstream's to keep current, not ours.
   * - ``docs``
     - Whether the published site answers
     - A docs badge should report the docs, not a build step.

The two served from our own site are the interesting ones. They are written by
``docs/badges/publish_badges.py``, which runs inside ``full`` after ``coverage report`` has
already failed the build if coverage fell short -- so the number a badge shows is one the gate
accepted, and the badge cannot be greener than the build. The output is published with the
documentation and never committed, which means a badge with no run behind it goes visibly stale
rather than quietly continuing to display a figure nobody re-measured.

This replaced a set of committed SVGs. Those were generated because ``cdadt`` was private and
shields.io could not read it, and they carried the defect that argument was paying for: the
coverage badge reported ``fail_under``, the threshold the suite *enforces*, not the coverage a
run *achieved*. True, checkable, and one step removed from what a reader assumes it says.

Live badges are not self-checking, though, and that is what ``test_docs.py`` is now for. A URL
can name a renamed workflow, a TOML key that no longer exists, or a site path CI does not write,
and shields.io renders every one of those as a small grey error that reads like a rendering
glitch. So each badge is traced offline to its source, an unrecognised badge shape *fails* rather
than passing unexamined, and the publisher is run against a fabricated coverage report to check
that its filenames are the ones the README asks for. One test exists purely to stop the coverage
figure rounding 99.96% up into a claim of ``100%``.

Continuous integration
----------------------

``.github/workflows/ci.yml`` has three jobs. Two of them are gates, split because the suite
divides very unevenly and pretending otherwise makes every push expensive; the third publishes
the documentation:

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 36

   * - Tests
     - Count
     - Time
     - Needs
   * - ``unit`` + ``contract``
     - 253
     - ~12 s
     - pip only
   * - ``integration``, ``verification``, ``validation``, ``slow``
     - 124
     - ~24 min
     - conda, IPOPT, JAX

Two thirds of the tests cost about a thousandth of the time.

``quick`` runs on **every push, on any branch**. Ubuntu, pip, no conda: the fast tests never
reach pyOptSparse, IPOPT, JAX or a vortex lattice, so none of the compiled stack has to exist
for them, which is the whole reason the job finishes in a couple of minutes. It runs ``black``,
``ruff``, the fast tests, and the documentation build. Note that ``matplotlib`` is installed
despite being an extra -- three artefact tests draw real figures and *fail* rather than skip
without it.

**The site this documentation builds into is published at**
https://realjoey.github.io/cdadt/. Pages is configured with GitHub Actions as its source rather
than a branch, so nothing is ever committed to a ``gh-pages`` branch and no Jekyll pass runs over
Sphinx's output.

The ``pages`` job builds nothing itself. It joins two things and deploys them: the rendered site
from ``quick``, and the badge documents from ``full``. Needing both is what makes the deployment
mean something -- a commit cannot replace the live site unless ``black``, ``ruff``, the *whole*
suite, the coverage threshold and ``sphinx -W`` all passed on it, so a red build anywhere leaves
the previous site standing.

It also costs something, and the cost is latency: the site trails a push to ``main`` by however
long ``full`` takes, up to about an hour. That is the price of a coverage badge reporting a
measured number, since only ``full`` measures it. Splitting this into a fast prose-only deploy
followed by a badge-only one would buy the latency back, at the cost of two deployments racing
for the same site.

**The rendered site is also downloadable from every run**, on any branch, as an artifact named
``documentation`` kept for 90 days. On a branch that artifact is the only way to see what the
build produced, since the deployed site only ever shows ``main``.

``full`` is the real gate: every test, the 100% coverage threshold, and ``sphinx -W``, in the
exact ``cdadt_env`` a developer uses. It runs on ``main``, on every pull request, and on request
-- not on a branch push, where ``quick`` is what guards you. A contract test asserts the workflow
still contains all of those commands, because a workflow trimmed to keep the runner cheap shows
the same green tick while standing for less.

A green ``quick`` is a statement about the fast tests only. It excludes everything that converges
a mission, so it cannot enforce the coverage threshold. The badge tracks ``main``, where both
jobs run.

Three further decisions in ``full`` are worth knowing:

**It runs on Windows.** ``environment.yml`` is an exact export of a validated Windows
environment; it names ``vc14_runtime``, ``ucrt``, ``win_inet_pton`` and ``pyside6``, none of
which exist for linux-64. Running Linux would mean a second environment specification -- a second
thing to keep true -- and would lose the ``=*openblas`` build selectors that file documents as
load-bearing precisely on Windows.

**The three dependencies are cloned at pinned commits**, not tracked. The validation tests assert
agreement with OpenConcept's own published example to 1e-6, so following upstream would let a
commit in somebody else's repository turn this badge red without a change here. Bumping a pin is
a deliberate act, and it comes with revalidating the numbers in :doc:`validation`.

**Everything is installed with** ``--no-deps`` **except OpenAeroStruct.** The reasons are the
same ones :doc:`install` gives for a developer's machine, and cdadt itself is included: letting
pip resolve the ``dev`` extra would install its own ``black``, and a ``black`` that differs from
the one in ``cdadt_env`` formats different files. CI and a developer would then disagree about a
gate whose whole purpose is to end that disagreement.

Actions minutes are free on a public repository, so the split no longer buys an allowance -- but
it still buys the wall clock, which is what it was really for. A branch push costs about two
Linux minutes instead of about ninety Windows ones.
Superseded runs of the same ref cancel, and ``full`` carries a 150-minute ceiling against
GitHub's six-hour default.

``full`` also runs **weekly**, at 06:00 UTC on Sundays. Nothing in the repository changes between
those runs, which is the point: conda's solver, PyPI, the runner image and three pinned upstream
clones all can, and a suite that passes only until the world moves underneath it is worth
learning about on a schedule rather than from whoever pushes next.

Security scanning, and what is not here
---------------------------------------

The obvious thing to reach for is GitHub's CodeQL. Until this repository was made public it could
not run at all -- code scanning on a **private** repository requires GitHub Advanced Security, and
``/repos/realJoey/cdadt/code-scanning/alerts`` returned 403 without it. That is no longer the
constraint. The same endpoint now answers 404 *"no analysis found"*, which is the difference
between forbidden and simply never run. **CodeQL is therefore an open decision, not a closed
one**, and this section describes what is in place rather than what is possible.

The dependency-review action is in the same position with one measured caveat: it reads the
dependency graph, and as of 2026-08-03 ``/repos/realJoey/cdadt/dependency-graph/sbom`` still
answers 404 for this repository while the same call against a public ``mdolab/openconcept``
returns a populated SBOM. So the graph is not there to read yet, whatever the reason.

Adding either workflow before that is settled would produce permanently failing checks, and that
is not a theoretical concern: the sibling ``penguino`` repository carries both, and they have
failed **39 and 11 times respectively, without a single success**. A check that is always red
teaches you to ignore red, which costs more than the check was ever worth.

What is actually in place, and why it was worth having even when CodeQL was unavailable:

**Static security analysis runs in the linter.** ``ruff``'s ``S`` rules are flake8-bandit, and
they now run on every push in a job that takes two minutes. The ``cdadt`` package passes them
with nothing ignored. The exemptions in ``per-file-ignores`` are scoped to ``tests/`` and
``docs/`` and are exactly two ideas: ``assert`` is what a test is, and two files legitimately
shell out -- the boundary test to ``git``, the XDSM builder to a LaTeX engine -- on arguments
they construct themselves.

**Dependency freshness comes from Dependabot**, in ``.github/dependabot.yml``. Version updates
need no dependency graph, which is why they worked when the repository was private and why they
are unaffected by its now being public. It watches the GitHub Actions and pip ecosystems monthly.
This addresses a failure that really happened: the workflow was first written against
``actions/checkout@v4``, ``actions/cache@v4`` and ``setup-miniconda@v3`` when the current majors
were v7, v6 and v4, and nothing in the repository would have said so.

The three research dependencies are deliberately **excluded** from Dependabot. They are pinned to
exact commits because the validation tests assert agreement with OpenConcept's published example
to 1e-6; a bot bumping them would be proposing to invalidate published numbers, which is a
decision for whoever re-runs them.

Branch protection
-----------------

Tiered CI is only a gate if something enforces it. Without protection on ``main``, ``full``
running there is a *report* -- it says the suite broke after ``main`` already moved, which is the
wrong way round.

``main`` should therefore require both ``quick`` and ``full`` to pass, through a pull request,
with linear history and no force pushes:

.. code-block:: bash

   gh api -X PUT repos/realJoey/cdadt/branches/main/protection --input protection.json

The consequence is a real change in workflow: ``main`` can no longer be moved by a local
fast-forward, which is how it has been updated until now. Work goes to ``dev``, a pull request
opens against ``main``, and ``full`` gates the merge. Self-merging is allowed -- the rule requires
a pull request, not a second reviewer.
