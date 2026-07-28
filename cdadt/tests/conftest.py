"""Pytest configuration for the cdadt test suite.

This module runs before any test module imports OpenMDAO, which is the only point at which
OpenMDAO's reporting system can be configured. Reports are disabled for the suite: they
write ``<script>_out/`` directories next to the working directory, which is noise during a
test run and would otherwise be mistaken for build output.

Nothing here changes model behavior. Report generation is a side output only; disabling it
does not alter any computed result.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")
