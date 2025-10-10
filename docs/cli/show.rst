gamspy show
===========

Shows information about your GAMSPy installation.

Show License
------------

Shows the content of the current license.

Usage
~~~~~

::

  gamspy show license

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-verbose
     - -v
     - False
     - Show verbose information about the license.

Example::

  $ gamspy show license
  License found at: /home/user/.gamspy/gamspy_license.txt

  License Content
  ===============
  [License content will be displayed here]

  License expiration date: xxxx-xx-xx

  License type: <license_type>

  Licensed Solvers:
  [List of licensed solvers goes here]



Show Base Directory
-------------------

Shows the path of the gamspy_base installation directory.

Usage
~~~~~

::

  gamspy show base

Example::

  $ gamspy show base
  /home/user/miniconda3/envs/gamspy/lib/python3.9/site-packages/gamspy_base
