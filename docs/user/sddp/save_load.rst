.. _sddp_save_load:

.. meta::
   :description: Saving and loading a trained SDDP policy in GAMSPy
   :keywords: SDDP, save, load, persistence, sddp file, serialize, GAMSPy, gamspy, GAMS

******************
Saving and Loading
******************

A trained policy is just its cuts, so there is no need to retrain every session.
``save()`` writes a trained instance to a single file, and ``SDDP.load()`` brings
it back.

.. code-block:: python

   # after training
   sddp.save("clearlake.sddp")

   # later, in a fresh session
   from gamspy.formulations.sddp import SDDP

   sddp = SDDP.load(
       "clearlake.sddp",
       solver="cplex",
       solver_options={"threads": 1},
   )
   report = [sddp.container[name] for name in ("R", "L", "Z", "F")]
   sddp.policy(stage="mar", state=150, noise=350, report=report)

The path must end in ``.sddp``. The file bundles the host container (with all the
trained cuts) and a small sidecar recording which symbols play which role, so a
reloaded instance is ready to use immediately.

Solver configuration is not saved with the model when using ``save()``.
To reuse the same settings, pass ``solver`` and ``solver_options`` again
when loading. If omitted, the loaded policy uses the system's default LP
solver without custom solver options.

Read-only instances
===================

A loaded instance is **read-only**: ``policy()`` and ``simulate()`` work, but
``add_state()``, ``set_noise()``, ``build()`` and ``train()`` raise. To change the
model or add iterations, rebuild and retrain from scratch.

.. note::
   The file records the module version it was written with. A file from a
   different major version is rejected with a clear message rather than
   failing obscurely; retrain to produce a current file.

.. seealso::
   :doc:`policy_and_simulation` covers the queries a loaded policy supports; the
   :doc:`ClearLake tutorial <clearlake>` saves and reloads a policy end to end.
