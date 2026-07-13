.. _sddp_training:

.. meta::
   :description: Training an SDDP policy in GAMSPy: iterations, early stopping, the optimality gap
   :keywords: SDDP, train, convergence, early stopping, rel_tol, patience, gap_paths, optimality gap, GAMSPy, gamspy

************************
Training and Convergence
************************

``train()`` runs the SDDP forward/backward iterations (explained in
:doc:`how_it_works <how_it_works>`) and returns an ``SDDPResult`` holding the
lower bound and the convergence history.

.. code-block:: python

   result = sddp.train(n_iter=20, rel_tol=1e-3, patience=3)

Iterations and early stopping
=============================

``n_iter`` is the hard cap on the number of iterations. On its own the run does
exactly that many and reports ``stop_reason == "max_iter"``.

The lower bound usually plateaus well before the cap, so ``rel_tol`` and
``patience`` add a stopping rule: training stops once the bound improves by less
than ``rel_tol`` (relative) for ``patience`` consecutive iterations, reporting
``stop_reason == "converged"``. On ClearLake, ``rel_tol=1e-3, patience=3`` stops
after 8 of the 20 allowed iterations. Leaving ``rel_tol=None`` (the default)
disables early stopping and runs all ``n_iter``.

To train a risk-averse policy, pass ``risk=``; see :doc:`risk`.

The result
==========

``SDDPResult`` carries the outcome of the run:

- ``lower_bound``: the rigorous lower bound at the final iteration.
- ``iterations_run``: how many iterations actually ran.
- ``stop_reason``: ``"converged"``, ``"max_iter"`` or ``"interrupted"``.
- ``convergence_table``: the per-iteration bounds.

``print(result)`` prints the summary box shown in the
:doc:`tutorial <clearlake>`. When the instance is verbose (the default),
training also prints one row per iteration as it goes.

Measuring the optimality gap
============================

The lower bound tells you how good the policy *could* be, not how good it *is*.
Passing ``gap_paths`` runs an out-of-sample Monte Carlo of the trained policy
after training and reports an estimated gap:

.. code-block:: python

   result = sddp.train(n_iter=20, rel_tol=1e-3, patience=3, gap_paths=500)

.. code-block:: text

     Policy cost          :    1.210000E+2 ±  3.626809E+1   (500 MC paths, 95% CI)
     Optimality gap       :        7.1862 %

The true optimum sits between two computed values: the lower bound underneath
it and the expected cost of any feasible policy above it. The distance between
them is the optimality gap. The simulation *estimates* that expected cost from
``gap_paths`` random paths, so ``result.optimality_gap_pct`` is a Monte Carlo
estimate rather than a rigorous bound; the reported confidence interval
(``policy_cost_mean`` ± 1.96 × ``policy_cost_stderr``, the 95% interval shown
in the box) says how wide that estimate is. ``gap_paths=0`` (the default) skips
this entirely and is perf-neutral.

.. note::
   This gap is only meaningful for risk-neutral training. With
   ``risk=CVaR(...)`` the lower bound is the risk-adjusted cost while the
   simulation still reports the plain average cost, so the two do not line up
   and ``optimality_gap_pct`` is not an optimality gap; ignore it in that case.
   ``policy_cost_mean`` on its own still tells you the policy's average cost
   over the sampled paths, though not the tail cost that ``risk=CVaR(...)``
   trains against.

Interrupting training
=====================

Long runs can be stopped gracefully. Pressing :kbd:`Ctrl+C` once finishes the
current iteration, then returns the policy trained so far with
``stop_reason == "interrupted"``, and the cuts learned up to that point are intact
and usable. Pressing :kbd:`Ctrl+C` a second time aborts hard.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` shows a full training run and its
   summary; :doc:`how_it_works <how_it_works>` explains what each iteration
   does and why the lower bound can be trusted.
