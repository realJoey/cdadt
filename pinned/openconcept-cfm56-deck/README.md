# The pinned CFM56 engine deck

These three files are the fitted Kriging surrogate that OpenConcept's ``CFM56()`` engine deck
uses. They are a **pinned input to this repository's validation**, in the same sense as the three
dependency commit SHAs in ``.github/workflows/ci.yml`` -- except that, unlike a commit SHA, this
one cannot be fetched from anywhere.

## Why they have to travel with the repository

``openconcept/propulsion/cfm56.py`` builds the deck as an ``om.MetaModelUnStructuredComp`` whose
outputs use ``om.KrigingSurrogate(training_cache=.../cfm56*_trained.zip)``. The surrogate is
fitted from the ``.npy`` data OpenConcept ships, and the fit is cached to those zips on first use.

The zips are not distributed. In the OpenConcept clone, ``git log --all -- '*_trained.zip'`` is
empty, ``.gitignore`` excludes ``*.zip``, and ``setup.py``'s ``package_data`` ships only
``["**/*.npy"]`` -- so neither a git clone nor a PyPI install provides one. Every machine fits its
own, once, and OpenConcept never invalidates it afterwards.

That fit is a SciPy SLSQP solve over the Kriging hyperparameters, so it follows the
floating-point path of whichever machine performs it. Measured 2026-08-07 by a fingerprint step
run on a GitHub Windows runner: fitting the thrust surrogate took 8 SLSQP iterations on the
developer machine and 6 on the runner, and the two resulting thrust surfaces differ by up to
**3.0e-5** relative. Two cold fits on two machines do not agree, so refitting cannot make this
reproducible. It can only be transported.

## What that difference does

A 3.0e-5 difference in the deck moves the converged baseline aeroplane by 8.1e-6 in MTOW, before
any optimizer runs -- 78345.0201 kg on the developer machine against 78345.6513 kg on the runner.

That is harmless for the sizing validation, which asserts to OpenConcept's own
``PUBLISHED_TOLERANCE`` and passes on both machines. It is not harmless for
``cases/b738_optimization.yaml``: that study's optimum sits against the climb-throttle bound, and
from the runner's baseline IPOPT had not converged after 40 iterations and 264 objective
evaluations, where from the developer baseline it converges in 25 iterations and 46 evaluations.
The CI failures of 2026-08-05 and 2026-08-07 were both exactly that -- ``Inform -1, Maximum
Iterations Exceeded``, four of four constraints met, nothing violated, the driver cut off
mid-descent.

## Provenance

Fitted 2026-01-18 from the ``.npy`` training data at OpenConcept ``0d2adeb``, the commit
``ci.yml`` pins. Both that data and ``cfm56.py`` are byte-identical between the commit the fit was
performed against and ``0d2adeb``, so these zips are a fit of the pinned data by the pinned code.

Every number in ``docs/validation.rst``, every run under ``run_outputs/``, and the converged
optima of all three optimization cases were produced against this deck.

## Bumping it

Replacing these is a deliberate act carrying the same weight as bumping a dependency pin: it
changes the engine every case in this repository flies, and it requires revalidating the
published numbers. To regenerate, delete the three zips from the OpenConcept clone, run any case
that touches the deck, copy the regenerated zips back here, and refresh the checksums with
``sha256sum *.zip > SHA256SUMS``.

## Verification

``SHA256SUMS`` is checked in CI immediately after the files are copied into the runner's
OpenConcept clone, before OpenConcept is installed. A silent mismatch would mean the whole suite
flew an engine nobody validated while printing entirely reasonable numbers, so that check is a
gate rather than a diagnostic.
