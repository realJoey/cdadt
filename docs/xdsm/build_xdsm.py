"""Draw an XDSM for each of the three case sets cdadt ships.

An XDSM (Lambe and Martins, 2012) says two things at once that a block diagram cannot: which
components exist and in what order they are executed, and which variables pass between them. That
second half is why it is worth having here. The whole point of cdadt's aerodynamics layer is that
the mission hands the lift coefficient *in* and takes drag *out*, and no prose diagram makes that as
plain as seeing ``fltcond|CL`` on a connection pointing at the loads model rather than away from it.

Three diagrams, one per set:

``xdsm_baseline``
    The aircraft configuration driven into OpenConcept's own analysis. Every drag number is
    OpenConcept's.
``xdsm_openavl``
    The same, with cdadt's loads component installed in the aircraft model and an openavl vortex
    lattice behind it.
``xdsm_openaerostruct``
    The same again, through OpenConcept's own OpenAeroStruct lattice.

Why the configuration is read rather than written down
-------------------------------------------------------

A diagram that disagrees with the code is worse than no diagram, and a hand-drawn one has no way to
notice. So each figure's title is built from the case file it depicts -- the analysis group, the
loads model, whether wave drag is on -- read out of the YAML at build time. The *topology* is still
authored here, but nothing in the caption can quietly drift away from what the study actually runs.

Output
------

pyXDSM emits LaTeX/TikZ and shells out to ``pdflatex``. Where there is no LaTeX the ``.tex`` is
still the artefact -- it is the source, and it is what a thesis would ``\\input`` -- so the build
degrades to writing that and saying so, rather than failing or pretending a PDF exists.

Run from anywhere::

    python docs/xdsm/build_xdsm.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from pyxdsm.XDSM import FUNC, LEFT, OPT, SOLVER, XDSM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CASES = ROOT / "cases"


class CaseUnderDiagram:
    """The configuration a diagram depicts, read from the case file rather than assumed.

    Parameters
    ----------
    sizing, optimization : str
        File names in ``cases/``. Both are read: an XDSM shows the optimizer, so the design
        variables and the objective come from the optimization case, while the analysis case is
        what confirms the two agree about the aerodynamics.

    Raises
    ------
    FileNotFoundError
        If either case file is missing, which means the diagram would depict something that no
        longer ships.
    """

    __slots__ = ("_optimization", "_sizing")

    def __init__(self, sizing: str, optimization: str) -> None:
        self._sizing = self._read(sizing)
        self._optimization = self._read(optimization)

    @staticmethod
    def _read(name: str) -> dict[str, Any]:
        """Load one case file, with comments stripped by the YAML parser."""
        path = CASES / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist; the diagram would depict a case that does not ship")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @property
    def analysis_group(self) -> str:
        """Short name of the analysis group being driven."""
        return self._sizing["black_box"]["model"].split(":")[-1]

    @property
    def loads_model(self) -> str | None:
        """Short name of the installed loads model, or None when the box supplies its own."""
        named = self._sizing["black_box"].get("options", {}).get("aerodynamic_loads")
        return named.split(":")[-1] if named else None

    @property
    def wave_drag(self) -> bool:
        """Whether transonic drag rise is installed."""
        return bool(self._sizing["black_box"].get("options", {}).get("wave_drag", False))

    @property
    def design_variables(self) -> list[str]:
        """The ``ac|`` names the optimization case frees, in declaration order."""
        return [name for name, spec in self._optimization["design_variables"].items() if "optimize" in spec]

    def caption(self) -> str:
        """A one-line description of what this diagram depicts, from the files themselves."""
        aerodynamics = self.loads_model or "the analysis group's own polar"
        drag_rise = "with transonic drag rise" if self.wave_drag else "no transonic drag rise"
        return f"{self.analysis_group}, {aerodynamics}, {drag_rise}"


class SizingDiagram:
    """One XDSM: the optimizer, the Newton-converged box, and what passes between them.

    The layout is the same for all three sets, which is the point -- only the aerodynamics block
    changes, and a reader comparing two figures should see one difference rather than two drawings.

    Parameters
    ----------
    case : CaseUnderDiagram
        Supplies the caption and the design variables, so those cannot disagree with ``cases/``.
    lattice : str or None
        Label for the vortex-lattice block, or ``None`` for the baseline, where the drag polar is
        inside the box and there is nothing of cdadt's to draw.
    """

    __slots__ = ("_case", "_lattice")

    def __init__(self, case: CaseUnderDiagram, lattice: str | None = None) -> None:
        self._case = case
        self._lattice = lattice

    def build(self) -> XDSM:
        """Assemble the diagram."""
        diagram = XDSM(use_sfmath=True)

        diagram.add_system("opt", OPT, r"\text{IPOPT}")
        diagram.add_system("cdadt", FUNC, (r"\text{cdadt}", r"\text{disciplines}"))
        diagram.add_system("newton", SOLVER, (r"\text{Newton}", r"\text{weight closure}"))
        diagram.add_system("geom", FUNC, (r"\text{OpenConcept}", r"\text{geometry, tails}"))
        diagram.add_system("weights", FUNC, (r"\text{OpenConcept}", r"\text{empty weight}"))

        if self._lattice is not None:
            diagram.add_system("loads", FUNC, (r"\text{cdadt}", r"\text{AerodynamicLoadsComp}"))
            diagram.add_system("lattice", FUNC, (r"\text{" + self._lattice + r"}", r"\text{fitted polar}"))
        diagram.add_system("mission", FUNC, (r"\text{OpenConcept}", r"\text{FullMissionWithReserve}"))
        diagram.add_system("results", FUNC, (r"\text{cdadt}", r"\text{SizingResults}"))

        # -- what the optimizer moves, and what it reads back ---------------------------------
        diagram.connect("opt", "cdadt", r"S_{ref}, A\!R, \Lambda_{c/4}, \lambda, T")
        diagram.connect("cdadt", "newton", r"\texttt{ac|}\text{ variables}")
        diagram.connect("results", "opt", r"W_{fuel}, \text{constraints}")

        # -- inside the box -------------------------------------------------------------------
        diagram.connect("newton", "geom", r"S_{ref}, A\!R, \lambda")
        diagram.connect("geom", "weights", r"\text{MAC}, S_{wet}, S_{h}, S_{v}")
        diagram.connect("weights", "mission", r"\text{OEW}")
        diagram.connect("newton", "mission", r"\text{MTOW}")
        diagram.connect("mission", "newton", r"W_{fuel}")

        if self._lattice is not None:
            # The direction here is the whole reason for drawing this: lift goes IN.
            diagram.connect("mission", "loads", r"C_L, q, M, h")
            diagram.connect("geom", "loads", r"S_{ref}, A\!R, \Lambda_{c/4}, \lambda")
            diagram.connect("loads", "lattice", r"\text{planform}")
            diagram.connect("lattice", "loads", r"C_{D_{min}}, k, C_{L_{minD}}")
            diagram.connect("loads", "mission", r"D")
        diagram.connect("mission", "results", r"\text{MTOW, fuel, BFL,}\ V_1")

        diagram.add_input("cdadt", r"\text{case file}")
        diagram.add_output("results", r"\text{report, JSON, figures}", side=LEFT)
        return diagram


#: The three sets, in the order the documentation introduces them.
DIAGRAMS = (
    ("xdsm_baseline", "b738.yaml", "b738_optimization.yaml", None),
    ("xdsm_openavl", "b738_avl.yaml", "b738_avl_optimization.yaml", "openavl"),
    ("xdsm_openaerostruct", "b738_oas.yaml", "b738_oas_optimization.yaml", "OpenAeroStruct"),
)


def main() -> int:
    """Write one XDSM per set, and report honestly whether a PDF could be produced."""
    latex = shutil.which("pdflatex")
    if latex is None:
        print("no pdflatex on PATH: writing .tex only, which is the source a thesis would input")

    for name, sizing, optimization, lattice in DIAGRAMS:
        case = CaseUnderDiagram(sizing, optimization)
        diagram = SizingDiagram(case, lattice).build()
        diagram.write(str(HERE / name), build=latex is not None, cleanup=latex is not None)
        freed = ", ".join(variable.split("|")[-1] for variable in case.design_variables)
        print(f"{name:<22} {case.caption()}")
        print(f"{'':<22} frees: {freed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
