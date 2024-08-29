gamspy probe
============

Probes the node (computer) to get information about the node for fingerprinting the license.

Usage
-----

::

  gamspy probe -j info.json  

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-json-out 
     - -j
     - 
     - Output path to dump probed information.

Example: ::

  $ gamspy probe -o info.json
  {
    "cpu_id": "27197016915918185882701231384169",
    "device_id": "18113801",
    "docker_mac_address": "",
    "hostname": "my_computer",
    "logical_cpu_cores": 8,
    "mac_addresses": [
      "00:15:5d:02:2f:76"
    ],
    "machine_id": "535bd6adab9ab283e853050532b042a9",
    "operating_system": "Linux",
    "physical_cpu_cores": 4,
    "running_in_docker": false,
    "running_in_kubernetes": false,
    "total_memory": 16,
    "username": "joe"
  }

.. note::
    The probed information is always written to standard output. The ``-o`` option will write a file in addition.