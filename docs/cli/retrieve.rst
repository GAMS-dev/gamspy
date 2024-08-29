.. _gamspy_retrieve:

gamspy retrieve
===============

Retrieves the license based on given probed information.

Usage
-----

::

  gamspy retrieve license <license_id> [-i <probed_info>] [-o <license_name>]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-input 
     - -i
     - 
     - Input path to the file with probed node information, potentially from a different node not connected to the internet/
   * - -\-output 
     - -o
     - standard output
     - Output path to write the license file.

Example: ::

  $ gamspy retrieve license 876e5812-1222-4aba-819d-e1e91b7e2f52
  Joe____________________________________________G240827+0003Ac-GEN
  joe@my.mail.com__________________________________________________
  07CPMK___________________________________________________________
  0COCOC___________________________________________________________
  CLA100251_876e5812-1222-4aba-819d-e1e91b7e2f52_O_FREEACADEMIC____
  node:18113801____________________________________________________
  MEYCIQDXZ42fd7G8MCppt6NXluallrcGdSiZRqFg9gbPxYBq1QIhAIZ7SvetdxRGj
  U0Piwc6zVAc0d/2pjm3iM70/mWToOSl__________________________________

::

  $ gamspy retrieve license 876e5812-1222-4aba-819d-e1e91b7e2f52 -i info.json -o gamslice.txt 
  Joe____________________________________________G240827+0003Ac-GEN
  joe@my.mail.com__________________________________________________
  07CPMK___________________________________________________________
  0COCOC___________________________________________________________
  CLA100251_876e5812-1222-4aba-819d-e1e91b7e2f52_O_FREEACADEMIC____
  node:18113801____________________________________________________
  MEYCIQDXZ42fd7G8MCppt6NXluallrcGdSiZRqFg9gbPxYBq1QIhAIZ7SvetdxRGj
  U0Piwc6zVAc0d/2pjm3iM70/mWToOSl__________________________________

.. note::
    The CLI tool ``gamspy retrieve license`` works together with ``gamspy probe`` and ``gamspy install license``. It's main purpose is to get a license
    for a node (or machine or computer) that is not connected to the internet and not capable of reaching ``license.gams.com`` to retrieve the
    license itself. In this case one runs ``gamspy probe -o info.json`` on the machine not connected to the internet, let's call this machine A.
    Now, we bring the file ``info.json`` to a machine connected to the internet, let's call this machine B. On machine B, one runs now 
    ``gamspy retrieve license -i info.json -o gamslice.A``. Now we bring the file ``gamslice.A`` to machine A and run on machine A 
    ``gams install license /path/to/gamslice.A``.