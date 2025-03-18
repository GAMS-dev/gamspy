.. _installation:

==========================
Installation and Licensing
==========================

Basic Installation
------------------

Creating a virtual environment is highly recommended whenever you start working on a new Python project.
You can create a virtual environment and activate it as follows:

.. tab-set-code::

    .. code-block:: Linux-MacOS

        python -m venv .gamspy_venv
        source .gamspy_venv/bin/activate

    .. code-block:: Windows(Bash)

        python -m venv .gamspy_venv
        source .gamspy_venv/Scripts/activate

    .. code-block:: Windows(PowerShell)

        python -m venv .gamspy_venv
        .gamspy_venv\Scripts\Activate.ps1

You can run the following command to install GAMSPy from PyPI::

    pip install gamspy

Licensing
---------
GAMSPy comes with a free demo license which lets you generate and solve small models.
For more information about GAMS licenses and how to get a new license, check 
`GAMS Licensing <https://www.gams.com/sales/licensing>`_.

Installing or updating your license
===================================
A GAMSPy license is a either an ASCII file of six lines or 36 character access code. 
In order to install your license, all you need to do is to run: ::

    gamspy install license <path_to_ascii_file or access code>

You can run: ::

    gamspy show license

to verify the installation of the license.

For machines that are not connected to the internet and a license specified by an access code, you can probe the node's data
and get a license via a machine connected to the internet. Details about this can be found below and in :ref:`gamspy_retrieve`.

.. note::
    
    GAMS and GAMSPy licenses are different, which means one cannot use an existing GAMS license for GAMSPy.
    We provide GAMSPy licenses for free if you already have a maintained professional GAMS license. 
    Contact sales@gams.com (with your GAMS license) to arrange for the delivery of a GAMSPy license.

.. note::
    
    GAMSPy is free for academics. Please check `GAMS Academic Program <https://www.gams.com/academics/>`_ for details.


Uninstalling your license
=========================
If you no longer wish to use your license, you can uninstall it with the following command: ::

    gamspy uninstall license

The demo license originally shipped with GAMSPy is reinstated.


License installation for offline machines
=========================================

In order to use GAMSPy in a machine which does not have an internet connection (offline), the license installation process
is as follows:

1. Generate a json file which contains the node information as follows: ::

    gamspy probe -o info.json

2. Move info.json file to a machine which has an internet connection and run: ::

    gamspy retrieve license <access code> -i info.json -o license.txt

3. Move license.txt to the machine that does not have an internet connection and run: ::

    gamspy install license license.txt   


Solvers
-------

GAMSPy comes with default solvers, and add-on solvers can be installed on demand.

Listing solvers
===============

To list the installed solvers on your machine, you can run::

    gamspy list solvers

Alternatively, if you want to list all available solvers, you can run the following::

    gamspy list solvers --all
    
The same information can also be accessed programmatically via the ``utils`` module of GAMSPy::
    
    import gamspy as gp
    import gamspy_base
    print(gp.utils.getInstalledSolvers(gamspy_base.directory))
    print(gp.utils.getAvailableSolvers())

.. note::
    All available solver packages can also be found on `PyPI <https://pypi.org/user/GAMS_Development>`_.


Installing/Uninstalling add-on solvers
======================================

The following command can be used to install add-on solvers: ::

    gamspy install solver <solver_name1> <solver_name2> ......

Similarly, an add-on solver can be uninstalled using: ::

    gamspy uninstall solver <solver_name1> <solver_name2> ......

If you want to install all add-on solvers, you can do by running: ::

    gamspy install solver --install-all-solvers

You can uninstall all add-on solvers in the same way by running: ::

    gamspy uninstall solver --uninstall-all-solvers

One can also recover the add-on solvers that they have installed in a previous GAMSPy version with: ::

    gamspy install solver --existing-solvers

.. note::
    
    To use an add-on solver to solve your model, remember to specify the ``solver`` argument 
    in the ``model.solve``. For example,

        your model definition

        ...

        ...

        model.solve(solver="xpress")

Updating GAMSPy
===============

``pip install gamspy`` implicitly upgrades the dependencies of GAMSPy (i.e. gamspy_base and gamsapi). 
Hence, if there is a new version of ``gamspy_base``, you need to reinstall the add-on solvers after an upgrade: ::

    pip install gamspy --upgrade
    gamspy install solver mosek conopt xpress
    # or 
    gamspy install solver --existing-solvers

Additional steps when using solver link licenses
================================================

Attempting to use a solver with a link license only you might need to perform additional steps to make
your solver license known to GAMSPy. For example, a GAMS/Gurobi-Link license but without a 
properly set up Gurobi license will result in a licensing error with a message describing 
the problem. To make the GAMS/Gurobi-Link work you do not need to download or install the 
Gurobi software but only your Gurobi license. 

You only need to set the ``GRB_LICENSE_FILE`` environment variable to the path of the Gurobi 
license (gurobi.lic) that you generated using the ``grbgetkey`` program::

    export GRB_LICENSE_FILE=/path/to/gurobi.lic
    
Similar instructions can be found in the `GAMS Solver Manual <https://www.gams.com/latest/docs/S_MAIN.html>`_ for solvers that offer link licenses.

Building From Source
--------------------

If you are a macOS or Linux user (or using a subsystem like WSL 
on Windows) familiar with the command line, you can build GAMSPy 
locally by following the instructions below.

Prerequisites
=============

Building GAMSPy requires the following software to be installed:

1) Python 3.9.x or newer

2) The GAMSPy source code::
    
        git clone git@github.com:GAMS-dev/gamspy.git

Installation
============

You can install GAMSPy from source using the following command::

    pip install .

Testing
-------

Tests have several markers such as unit tests, integration tests, and doc tests. 
The tests can be run with adding desired markers to the pytest command below. 
For example, unit tests and integrations tests can be run with: ::

    pytest -m 'unit or integration' tests

.. note::
    To see all markers, one can run `pytest --markers`.