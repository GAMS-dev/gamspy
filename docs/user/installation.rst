.. _installation:

==========================
Installation and Licensing
==========================

Basic Installation
------------------

You can run the following command to install GAMSPy from PyPI::

    pip install gamspy

Licensing
---------
GAMSPy comes with a free demo license which lets you generate and solve small models.
For more information about GAMS licenses and how to get a new license, check 
`GAMS Licensing <https://www.gams.com/sales/licensing>`_.

Installing or updating your license
===================================
A GAMSPy license is a either an ASCII file of six lines or 36 character identification number sent to you via e-mail. 
In order to install your license, all you need to do is to run::

    gamspy install license <path_to_ascii_file or identification number>

For machines that are not connected to the internet, you can install a license with the node information by running::

    gamspy install license -i <your_license_number>

This option is available only for licenses in identification number format.

.. note::
    
    GAMS and GAMSPy licenses are different, which means one cannot use an existing GAMS license for GAMSPy.
    We provide GAMSPy licenses for free if you already have a full GAMS license. 
    Contact sales@gams.com (with your GAMS license) to arrange for the delivery of a GAMSPy license.
    If you were using a GAMS license for GAMSPy before v0.13.0, please contact sales@gams.com to arrange
    a GAMSPy license.


Uninstalling your license
=========================
If you no longer wish to use your license, you can uninstall it with the following command: ::

    gamspy uninstall license

License installation for offline machines
=========================================

In order to use GAMSPy in a machine which does not have an internet connection (offline), the license installation process
is as follows:

1. Generate a json file which contains the node information as follows: ::

    gamspy probe -o info.json

2. Move info.json file to a machine which has an internet connection and run: ::

    gamspy retrieve license <your_license_id> -i info.json -o license.txt

3. Move license.txt to the machine that does not have an internet connection and run: ::

    gamspy install license license.txt   

GAMS/Gurobi-Link
================
Attempting to use the GAMS/Gurobi solver with a GAMS/Gurobi-Link license but without a 
properly set up Gurobi license will result in a licensing error with a message describing 
the problem. To make the GAMS/Gurobi-Link work you do not need to download or install the 
Gurobi software but only your Gurobi license. 

You only need to set the ``GRB_LICENSE_FILE`` environment variable to the path of the Gurobi 
license (gurobi.lic) that you generated using the ``grbgetkey`` program::

    export GRB_LICENSE_FILE=/path/to/gurobi.lic
    
Then, you can run the gamspy command as usual::

    gamspy install license <path_to_your_license_file>

.. note::
    
    To use Gurobi to solve your model, remember to specify the ``solver`` argument 
    in the ``model.solve``.

        your model definition

        ...

        ...

        model.solve(solver="gurobi")

Solvers
-------

GAMSPy comes with default solvers, and additional solvers can be installed on demand.

Listing Solvers
===============

To list the installed solvers on your machine, you can run either::

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


Installing/Uninstalling New Solvers
===================================

TThe following command can be used to install new solvers::

    gamspy install solver <solver_name>

Similarly, a solver can be uninstalled using::

    gamspy uninstall solver <solver_name>

Updating Solvers
================

If the solvers on your machine are not up-to-date, you can run::

    gamspy update

This will update all solvers to a version compatible with GAMSPy.

Building from source
--------------------

If you are a macOS or Linux user (or using a subsystem like WSL 
on Windows) familiar with the command line, you can build GAMSPy 
locally by following the instructions below.

Prerequisites
=============

Building GAMSPy requires the following software to be installed:

1) Python 3.8.x or newer

2) The GAMSPy source code::
    
        git clone git@github.com:GAMS-dev/gamspy.git

Installation
============

You can install GAMSPy from source using the following command::

    pip install .

Testing
-------

Tests are classified into three categories: unit tests, integration tests, and doc tests. The tests can be run with::

    python tests/test_gamspy.py

.. note::
    By default, only unit tests are run. To enable integration tests, the ``--integration`` argument should be provided.
    Doctests can be enable with the ``--doc`` argument.