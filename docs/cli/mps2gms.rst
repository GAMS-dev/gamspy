.. _gamspy_mps2gms:

gamspy mps2gms
==============

Translates an MPS or LP file into equivalent short generic GAMS and GAMSPy programs.
By default, this command writes a GAMSPy (``.py``) file and a GDX file.

Usage
-----

::

  gamspy mps2gms <input_file> [gdx_file] [gms_file] [OPTIONS]

.. note::
  If no output files are specified, the command automatically generates ``<input>.py`` and ``<input>.gdx``.

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - --py
     - 
     - None
     - Name of GAMSPy program output file.
   * - --dec
     - 
     - None
     - DEC file for specifying decomposition information.
   * - --columnintvarsarebinary
     - 
     - 0
     - Integer variables appearing first in COLUMNS section are binary (0, N, 1, Y).
   * - --duplicates
     - 
     - NOCHECK
     - Handle multiple coefficients in LP files (NOCHECK, ADD, IGNORE, ERROR).
   * - --orignames
     - 
     - NO
     - Make original names of columns/rows available (NO, MODIFIED, ALL).
   * - --stageshift
     - 
     - 2
     - Shift block numbers by this integer value for stage attributes.
   * - --convertsense
     - 
     - 0
     - Convert objective sense (0, N, 1, Y, MIN, -1, MAX).

Examples
--------

Translate using default behavior (creates ``model.py`` and ``model.gdx``)::

  $ gamspy mps2gms model.mps

Specify a custom GAMSPy output name::

  $ gamspy mps2gms model.lp --py custom_script.py

Include GAMS output explicitly::

  $ gamspy mps2gms model.mps model.gdx model.gms --py model.py