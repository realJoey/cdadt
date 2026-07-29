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

   pytest -q -m "not slow"      # the fast loop: 250 tests, about two minutes
   pytest -q                    # everything: 310 tests, about nine minutes
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
     - All 310 pass. A test that is slow is marked ``slow``, not deleted.
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

Run directories, ``__pycache__``, coverage output and built docs are all in ``.gitignore``.
Nothing under ``run_outputs/`` is ever committed: every file in it is regenerated by one command.
