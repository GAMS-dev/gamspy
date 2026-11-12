gamspy probe
============

Probes the node (computer) to get information about the node for fingerprinting the license.

Usage
~~~~~

::

  gamspy probe [OPTIONS]

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-json-out 
     - -j
     - None
     - Output path to dump probed information.
   * - -\-ports
     - -p
     - None
     - Starting port and number of ports to search for available ports for instantiating a use-limited license.
   * - -\-use-port
     - -u
     - None
     - Starting port for instantiating a use-limited license.
   * - -\-network-interfaces
     - -n
     - False
     - List all available network interfaces and their hashes instead of the node information.

Example
~~~~~~~

::

  $ gamspy probe -j info.json
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
    The probed information is always written to standard output. The ``-j`` option will write the information to a JSON file.