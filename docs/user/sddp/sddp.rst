.. _sddp:

*********************************
GAMSPy and Stochastic Programming
*********************************

.. meta::
   :description: Stochastic programming in GAMSPy with Stochastic Dual Dynamic Programming (SDDP)
   :keywords: stochastic programming, SDDP, multistage, GAMSPy, gamspy, GAMS, mathematical modeling, uncertainty

.. admonition:: Beta feature
   :class: caution

   SDDP has been added in GAMSPy 1.25.0 and is a **beta** feature under active development. Its public API
   may still change between releases without preserving backward compatibility,
   so pin your GAMSPy version for reproducible work and validate results before
   relying on them for real decisions. Feedback is very welcome on the
   `GAMS forum <https://forum.gams.com>`_.

Many decisions have to be made over time and under uncertainty: how much water
to release from a reservoir before the season's rainfall is known, how much to
store before tomorrow's demand, how to commit resources as the future unfolds.
GAMSPy supports these **multistage stochastic** problems through
:meth:`SDDP <gamspy.formulations.SDDP>` (Stochastic Dual Dynamic Programming).

You write the single-stage problem as an ordinary GAMSPy model, describe how the
state carries forward and how the uncertainty is distributed, and SDDP learns a
**policy**: a rule that, for any state and any realised outcome, returns the
optimal decision and its expected cost from there on.

When to use SDDP
================

SDDP fits problems that

* unfold over several **stages**, with decisions made before each stage's
  uncertainty is revealed;
* carry a **state** from one stage to the next (a reservoir level, an inventory,
  a stored quantity);
* have a convex (linear) stage problem and a finite set of noise **scenarios**.

If your problem is deterministic or has a single decision point, an ordinary
GAMSPy model is the simpler choice. For a hands-on start, the
:doc:`ClearLake tutorial <clearlake>` builds and trains a complete model in a
few dozen lines.

.. toctree::
   :maxdepth: 1

   ./introduction
   ./workflow
   ./state_variables
   ./scenarios
   ./risk
   ./training
   ./policy_and_simulation
   ./save_load
