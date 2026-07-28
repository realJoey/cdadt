.. _certification:

***********************
The certification basis
***********************

cdadt is certification-*driven*, not certification-*checked*. The difference is where the
regulations live. In a certification-checked method they are constraint expressions in a run
script, applied to a design that already exists. Here each one is an object that knows the
regulation it enforces, the model variables it reads, the components it contributes, the
constraint it registers with the optimizer, and how to report its own margin.

What a requirement is
=====================

A :class:`~cdadt.certification.basis.Requirement` declares:

* :attr:`~cdadt.certification.basis.Requirement.regulation` -- the citation, e.g.
  ``"14 CFR 25.121(b)"``;
* :attr:`~cdadt.certification.basis.Requirement.sense` -- whether the quantity must stay
  below a limit or above it;
* :meth:`~cdadt.certification.basis.Requirement.limit` -- read from configuration;
* :meth:`~cdadt.certification.basis.Requirement.constrained_path` -- *asked of the mission*,
  never hardcoded; and
* :meth:`~cdadt.certification.basis.Requirement.build` -- optional, for requirements
  covering conditions the mission does not model.

Requirements shipped
====================

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Regulation
     - Requirement
     - How it is evaluated
   * - **14 CFR 25.113**
     - Takeoff distance within the field available
     - OpenConcept already solves V\ :sub:`1` implicitly so the continue and abort distances
       match. cdadt applies the limit and records which runway it came from.
   * - **14 CFR 25.121(b)**
     - OEI second-segment climb gradient
     - Constrains OpenConcept's engine-out climb angle at V\ :sub:`2`. The regulation states
       a *gradient*; OpenConcept reports an *angle*. cdadt converts rather than conflating
       them.
   * - **14 CFR 25.125** with **121.195(b)**
     - Landing field length within the field available
     - New components. OpenConcept models no landing at all.
   * - Approach category
     - V\ :sub:`REF` within the category limit
     - Reads the same landing chain, so the reported approach speed and field length
       necessarily describe the same landing.
   * - *design constraint*
     - Throttle within limit, per phase
     - Labelled ``design``, not a regulation. One requirement per phase, so the matrix
       reports *which* phase is short of thrust.

Why the throttle constraint is in the certification basis
=========================================================

It is not a regulation, and the traceability matrix says so. It is there because it is what
makes the others meaningful.

OpenConcept's steady-flight phases solve for whatever throttle produces zero acceleration,
and that solve is perfectly happy to return 1.4. The mission then converges on an aircraft
flying at 140% of rated thrust -- and no certification requirement notices, because each one
reads a distance or a gradient that the over-thrusted aircraft achieves easily. Constraining
throttle is what turns "the mission converged" into "the engine can fly this mission".

Landing performance
===================

OpenConcept models no landing, so cdadt supplies it. The chain is:

``FlapCLmax`` at the landing flap setting → ``StallSpeed`` at maximum landing weight →
:class:`~cdadt.certification.components.landing.ApproachSpeed` →
:class:`~cdadt.certification.components.landing.LandingFieldLength`.

.. math::

   V_\text{REF} = k\, V_{SR0}, \qquad
   s_\text{landing} = s_\text{air} + \frac{V_\text{REF}^2}{2\bar{a}}, \qquad
   s_\text{field} = \frac{s_\text{landing}}{f_\text{dispatch}}

The two OpenConcept components in that chain are the ones already used elsewhere in the
model, so landing CLmax comes from the same correlation as takeoff CLmax and the two cannot
disagree about the same wing.

The method is deliberately transparent rather than an empirical fit. Every quantity in it is
one an engineer can defend or measure: how far the aircraft floats from the 50 ft screen
height, how hard it decelerates, and which operating rule applies. An empirical correlation
would hide all three inside a single coefficient calibrated on aircraft that are not the one
being designed.

**Where the physics lives.** The landing chain is a *provider*
(:class:`~cdadt.providers.openconcept.landing.LandingPerformanceProvider`), not part of the
requirement. A requirement states a limit and reads a value; if it also computed the value,
the regulation and the method would be entangled and swapping the landing method would mean
editing the regulation. This is enforced: ``cdadt/tests/mission/test_boundary.py`` fails the
suite if anything under ``cdadt/certification`` imports OpenConcept.

Every limit is configured
=========================

A field length or a climb gradient comes from a regulation **and an operating case** -- a
runway, an altitude, a temperature, an approach category. The regulation says what must be
demonstrated; the case says against what. A limit built into the code applies itself to
aircraft nobody chose it for, so cdadt requires both to be stated:

.. code-block:: yaml

   certification:
     climb:
       oei_second_segment_gradient:
         value: 0.024
         source: >-
           14 CFR 25.121(b): 2.4% steady gradient of climb at V2 with the critical engine
           inoperative, for two-engine aeroplanes. Three-engine is 2.7%, four-engine 3.0%.

The ``source`` is what the traceability matrix cites.

One limit is checked rather than merely read: an approach speed factor below the 1.23 floor
of §25.125(a)(2) is **rejected**, because it describes an aircraft that cannot be certified
and would produce landing distances shorter than any real aircraft achieves.

The traceability matrix
=======================

.. code-block:: text

         regulation  requirement                     value       limit      margin  status
   ----------------------------------------------------------------------------------------
   ICAO/FAA cat       Approach speed within cat    152.2540    140.0000    -12.2540  NOT MET kn
   14 CFR 25.125      Landing field length        7258.8578   7000.0000   -258.8578  NOT MET ft
   design             Throttle limit in climb        0.9408      1.0500      0.1092  MET
   14 CFR 25.113      Takeoff distance            5247.7076   8000.0000   2752.2924  MET ft
   14 CFR 25.121(b)   OEI climb gradient             0.0422      0.0240      0.0182  MET rad

Rows are ordered by *relative* margin, so the requirement actually driving the design appears
first. Absolute margins in feet, knots and radians are not comparable; relative ones are.

This is the artifact a certification-driven method has to produce. A converged design tells
you it closed. This tells you which requirements shaped it, by how much, and on whose
authority.

Verifying that a requirement is wired
=====================================

A requirement class that exists, cites the right regulation, and never registers a
constraint is worse than no requirement at all: it appears in the matrix and reports a
margin while the optimizer runs unconstrained. So the tests do not check that requirement
classes exist. They

* introspect the model's registered constraints and confirm each requirement appears;
* confirm each bound is on the correct side, since registering a field length as a *lower*
  bound would drive the optimizer to lengthen the runway requirement while every reported
  number still looked reasonable; and
* move each limit past the value the converged design achieved and confirm the requirement
  flips to not-met -- which proves it is reading the model rather than reporting a constant.
