.. _sddp_risk:

.. meta::
   :description: Risk-averse SDDP in GAMSPy with CVaR: tail, weight, and the risk-adjusted bound
   :keywords: SDDP, risk, CVaR, conditional value at risk, tail, weight, risk-averse, GAMSPy, gamspy, GAMS

********************
Risk Aversion (CVaR)
********************

By default SDDP minimises the **expected** total cost, which is the risk-neutral choice. But
the average is not always what you care about. The :doc:`ClearLake tutorial
<clearlake>` ends with a simulate summary that is cheap on most paths yet has an
expensive tail: a median of 0 but a 95th percentile of 1500. When the bad
outcomes are what matter, you can train a more conservative policy with
**Conditional Value-at-Risk (CVaR)**.

Conditional Value-at-Risk
=========================

For a cost distribution, :math:`\mathrm{CVaR}_\alpha` is the average cost over
the worst :math:`\alpha` fraction of outcomes. With :math:`\alpha = 0.25` it is
the mean of the costliest quarter of cases, a measure of the tail rather than
the centre.

The CVaR risk measure
=====================

Pass a :meth:`CVaR <gamspy.formulations.CVaR>` object to ``train()``:

.. code-block:: python

   from gamspy.formulations import CVaR

   result = sddp.train(n_iter=40, risk=CVaR(tail=0.25, weight=0.5))

``CVaR`` takes two numbers:

- ``tail``: the tail probability :math:`\alpha \in (0, 1]`, the worst fraction
  CVaR averages over.
- ``weight``: the risk weight :math:`w \in [0, 1]`, how much to lean on the
  tail versus the average.

At each stage the cost-to-go is aggregated as a blend of the two:

.. math::

   (1 - w)\,\mathbb{E}[Z] \;+\; w\,\mathrm{CVaR}_\alpha[Z],

where :math:`Z` is the stage cost plus future cost. With ``weight=0`` this is the
plain expectation (risk-neutral); with ``weight=1`` it is pure CVaR over the
tail; values in between trade the two off.

The risk-adjusted bound
=======================

A risk-averse policy guards against the tail, so it costs more on average, and
the lower bound reflects a different objective. Training ClearLake under CVaR
raises the bound well above the risk-neutral 112.30:

.. list-table::
   :header-rows: 1

   * - ``risk``
     - lower bound
   * - ``None`` (risk-neutral)
     - 112.30
   * - ``CVaR(tail=0.25, weight=0.5)``
     - 399.17
   * - ``CVaR(tail=0.25, weight=1.0)``
     - 916.67

The summary marks the bound as risk-adjusted, because it now bounds the risk
measure rather than the expected cost:

.. code-block:: text

   ========================================================================
     Risk measure         : CVaR(tail=0.25, weight=1.0)
     Lower bound (risk-adj.):    9.166667E+2
     Iterations run       :             40
     Stop reason          :       max_iter
   ========================================================================

These risk-adjusted bounds are not comparable to the risk-neutral one, since they
measure a deliberately more pessimistic objective.

.. note::
   SDDP uses the standard nested, **first-stage-neutral** convention: the first
   stage is risk-neutral (an expectation) and CVaR is applied to the later
   stages. The default ``risk=None`` is exactly the risk-neutral expectation,
   bit-for-bit. CVaR also composes with non-uniform
   :doc:`probabilities <scenarios>`, so the tail is taken with respect to your
   scenario weights.

.. seealso::
   The :doc:`ClearLake tutorial <clearlake>` shows the skewed cost distribution
   that motivates shaping the tail; :doc:`scenarios` covers the probabilities
   CVaR is measured against.
