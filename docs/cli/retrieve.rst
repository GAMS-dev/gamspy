.. _gamspy_retrieve:

gamspy retrieve
===============

Retrieves a license with another node's information.

Retrieve License
----------------

Retrieves a license using an access code and node information from a JSON file.

Usage
~~~~~

::

  gamspy retrieve license <access_code> [OPTIONS]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-input
     - -i
     - None
     - Input JSON file path to retrieve the license based on the node information.
   * - -\-output
     - -o
     - None
     - Output path for the license file.
   * - -\-checkout-duration
     - -c
     - None
     - Specifies a duration in hours to checkout a session.

Examples
~~~~~~~~

Retrieve a license with node information::

  $ gamspy retrieve license 876e5812-1222-4aba-819d-e1e91b7e2f52 --input node_info.json

Retrieve and save the license to a file::

  $ gamspy retrieve license 876e5812-1222-4aba-819d-e1e91b7e2f52 --input node_info.json --output license.txt

.. note::
    The input JSON file should contain the node information required for license retrieval.