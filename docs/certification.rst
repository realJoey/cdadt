The certification basis
=======================

This is what makes cdadt certification-*driven* rather than certification-*checked*.

A :class:`~cdadt.certification.Requirement` is an object. It knows the regulation it enforces,
the limit that regulation imposes, where that limit came from, the model quantity it
constrains, any analysis it must contribute to evaluate itself, and how to report its own
margin. A :class:`~cdadt.certification.CertificationBasis` is the set of them a design is
being certified against.

The requirements shipped
------------------------

===================================================== ======================== ================================
Class                                                 Citation                 Constrains
===================================================== ======================== ================================
:class:`~cdadt.certification.BalancedFieldLength`     14 CFR 25.113            ``mission.bfl.distance_continue``
:class:`~cdadt.certification.EngineOutClimbGradient`  14 CFR 25.121(b)         ``mission.engineoutclimb.gamma``
:class:`~cdadt.certification.ApproachSpeed`           14 CFR 25.125 / 97       V\ :sub:`ref` (contributed)
:class:`~cdadt.certification.ThrottleMargin`          design limit             ``mission.<phase>.throttle``
===================================================== ======================== ================================

Three of the four constrain a quantity the mission already produces. The fourth contributes
the analysis it needs, because the mission does not fly a landing.

Nothing is defaulted
--------------------

Every requirement takes its limit and the limit's ``source`` as constructor arguments, both
required::

    BalancedFieldLength(
        limit=8000.0,
        source="Design field length, 8000 ft dry runway at sea level, ISA",
    )

A field length or a required climb gradient comes from a regulation *and* an operating case --
a runway, an altitude, a temperature, a fleet category, an engine count. Building one into the
code applies it to aircraft nobody chose it for, and a limit with no recorded provenance is
indistinguishable from a guess in the report that quotes it.

Gradients are not angles
------------------------

§25.121(b) states a *gradient*: rise over run, a tangent. OpenConcept reports the climb
*angle*, in radians. :class:`~cdadt.certification.EngineOutClimbGradient` converts with
``arctan``, which is exactly equivalent because ``tan`` is monotonic over the range of
interest. For a 2.4% gradient the difference is under 0.1% of the value. Small -- but real,
and conflating the two is how small errors get into a certification argument.

Takeoff flaps for the second-segment climb
------------------------------------------

§25.121(b) specifies the gradient "with the landing gear retracted and the takeoff flaps set".
OpenConcept's own B738 example evaluates its engine-out climb condition with **clean** drag,
which is optimistic against that regulation.

That is a property of the model, not of the requirement, so cdadt fixes it in the model:
:class:`~cdadt.model.Part25PhaseModel` adds the engine-out climb condition to the set of
phases flown in takeoff configuration. It is opt-in, and the base model reproduces the
reference example exactly. On the B738 case the difference is large -- 0.058 rad clean against
0.042 rad with takeoff flaps -- which is exactly why it should not be silent. See
:doc:`validation`.

Constraint scaling
------------------

:meth:`~cdadt.certification.Requirement.register` scales every constraint by ``1 / |limit|``,
so each is order one when it is binding. This is not cosmetic. A certification basis spans a
field length of thousands of feet, an approach speed of a hundred-odd knots and a climb
gradient of a few hundredths of a radian. Presented unscaled, an optimizer sees the field
length as five orders of magnitude more important than the gradient and the gradient as
satisfied to within its own noise.

The approach-speed requirement
------------------------------

§25.125(a)(2), harmonized with §25.107(c), defines the landing distance from 50 ft at an
approach speed of at least 1.23 V\ :sub:`SR0`. That reference speed sorts an aeroplane into an
approach category -- ICAO Doc 8168 and 14 CFR 97 category C is 121-140 kn -- and the category
decides which aerodromes and approach procedures the type may use. It is an operational
requirement with regulatory teeth, and it bites on wing area.

The mission does not fly a landing, so the requirement contributes the analysis, built from
OpenConcept components only:

.. math::

   C_{L_{max,land}} = \mathrm{FlapCLmax}(\delta_\mathrm{land}),
   \qquad
   V_\mathrm{SR0} = \sqrt{\frac{2 W_\mathrm{MLW} g}{\rho_0 S_\mathrm{ref} C_{L_{max,land}}}},
   \qquad
   V_\mathrm{ref} = 1.23\, V_\mathrm{SR0}

It requires ``ac|aero|landing_flap_deg`` in the aircraft definition.

The traceability matrix
-----------------------

Every run against a basis emits one, sorted most-binding first:

.. code-block:: text

   regulation             requirement                                 value        limit       margin units  status
   ----------------------------------------------------------------------------------------------------------------
   design                 Throttle within limit in climb             1.0000       1.0000      -0.0000 -      ACTIVE
   14 CFR 25.125 / 97     Approach speed within category           136.4991     140.0000       3.5009 kn     MET
   design                 Throttle within limit in cruise            0.8287       1.0000       0.1713 -      MET
   14 CFR 25.113          Takeoff distance within field availa    6263.7145    8000.0000    1736.2855 ft     MET
   14 CFR 25.121(b)       OEI second-segment climb gradient          0.0406       0.0240       0.0166 rad    MET

   Limit sources:
     14 CFR 25.113: Design field length, 8000 ft dry runway at sea level, ISA
     14 CFR 25.121(b): 14 CFR 25.121(b)(1)(i), two-engine aeroplane
     14 CFR 25.125 / 97: ICAO Doc 8168 / 14 CFR 97 approach category C upper bound, 140 kn
     design: Engine deck rated condition; the CFM56 surrogate is not fitted above throttle 1

   5 of 5 requirements met (1 active, 0 not met).

Three details in that table are deliberate:

**ACTIVE is a status, not a failure.** A converged optimizer lands a binding constraint a hair
either side of its bound. Testing satisfaction by exact sign would report an active
requirement as violated by 10\ :sup:`-14` of a knot; the reporting tolerance is relative, and
it does not relax the constraint the optimizer actually enforced.

**Sorting is by relative margin.** 400 ft of field length and 0.0003 rad of climb gradient
cannot be compared as absolute margins. Sorted absolutely, the field length would always look
like the tightest requirement.

**Vector quantities report their worst element.** Throttle across a phase satisfies a
requirement only if it does so everywhere, so the peak is what appears.

The active requirements are the interesting result of an optimization: they are the ones that
shaped the design. A converged result alone does not say which those were.

What is not modeled
-------------------

Stated here with the same prominence as what is. cdadt does **not** currently model:

* **§25.125 landing field length.** The approach speed is constrained; the distance from 50 ft
  to a stop is not. Adding it means an empirical landing-roll correlation, which is exactly
  the kind of number this repository refuses to invent.
* **§25.119 balked landing climb** and **§25.121(a), (c), (d)** -- the first-segment,
  en-route and approach climb gradients. Only §25.121(b) is covered.
* **§25.107 V\ :sub:`MU`, V\ :sub:`MCG`, V\ :sub:`MCA`** and any lateral-directional
  requirement. The mission is a longitudinal point-mass model; there is no sideslip in it.
* **Structural, systems, flutter, or icing requirements** of any kind.

Reference
---------

.. automodule:: cdadt.certification
   :members:
   :noindex:
