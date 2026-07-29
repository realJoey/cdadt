"""Turning a ``"module.path:ClassName"`` string into the class it names.

A cdadt case file never imports anything. It *names* things -- the analysis to drive, the
aerodynamic loads model to install -- and cdadt resolves those names at run time. That is what
lets a study choose a black box or a physics model without a line of Python, and it is why no
cdadt module needs to import the analysis it drives.

The failure modes are all user errors in a text file, so each one is reported with what was
tried rather than as an import traceback from somewhere inside the dependency.
"""

from __future__ import annotations

import importlib

__all__ = ["ClassSpec"]


class ClassSpec:
    """A ``"module.path:ClassName"`` reference, and the class it resolves to.

    Parameters
    ----------
    spec : str
        The reference, as written in the case file.
    describes : str, optional
        What the reference is *for*, used in error messages -- ``"black-box model"``,
        ``"aerodynamic loads model"``. A message that names the key the reader wrote is worth
        more than a correct but anonymous one.

    Examples
    --------
    >>> ClassSpec("cdadt.models.polar:PolarLoads", describes="aerodynamic loads model").resolve()
    <class 'cdadt.models.polar.PolarLoads'>
    """

    __slots__ = ("_describes", "_spec")

    def __init__(self, spec: str, describes: str = "class") -> None:
        self._spec = str(spec)
        self._describes = str(describes)

    @property
    def spec(self) -> str:
        """The reference as written."""
        return self._spec

    def resolve(self, error: type[Exception] = ValueError) -> type:
        """Return the class this reference names.

        Parameters
        ----------
        error : type, optional
            Exception class to raise on failure. Each caller passes its own domain error, so a
            mistyped black box raises :class:`~cdadt.blackbox.BlackBoxError` and a mistyped loads
            model raises the loads error, rather than both raising something generic.

        Raises
        ------
        Exception
            Of type ``error``, naming what was tried: a reference without exactly one colon, a
            module that will not import, a missing attribute, or an attribute that is not a class.
        """
        if self._spec.count(":") != 1:
            raise error(f"The {self._describes} must be given as 'module.path:ClassName'; got {self._spec!r}.")
        module_name, class_name = self._spec.split(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as cause:
            raise error(f"Cannot import the module '{module_name}' named by the {self._describes}.") from cause
        try:
            candidate = getattr(module, class_name)
        except AttributeError as cause:
            raise error(f"'{module_name}' has no attribute '{class_name}'.") from cause
        if not isinstance(candidate, type):
            raise error(f"'{self._spec}' names a {type(candidate).__name__}, not a class.")
        return candidate

    def __repr__(self) -> str:
        """Return a representation naming the reference."""
        return f"ClassSpec({self._spec!r})"
