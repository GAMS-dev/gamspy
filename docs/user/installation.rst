.. _installation:

==========================
Installation and Licensing
==========================

Basic Installation
------------------

You can run the following command to install GamsPy from PyPI::

    pip install gamspy

Building from source
--------------------

If you are a MacOS or Linux user familiar with using the command line, 
you can continue with building GamsPy locally by following the instructions below.

Prerequisites
=============

Building GamsPy requires the following software installed:

1) Python 3.8.x or newer

2) The GamsPy source code::
    
        git clone git@github.com:GAMS-dev/gamspy.git

Installation
============

You can install GAMSPy from source with the following command::

    python setup.py sdist 
    pip install gamspy --find-links dist/

Licensing
---------
GAMSPy comes with a free demo license which lets you generate and solve small models.
For more information about GAMS licenses and how to get a new license, check 
`GAMS Licensing <https://www.gams.com/latest/docs/UG_License.html>`_.

Installing or updating your license
===================================
A GAMS license file is an ASCII file of six lines, which is sent to you via e-mail. 
Please copy all six lines into a file (typically named gamslice.txt). Then, run::

    gamspy install license <path_to_your_license_file>

Testing
-------

Tests can be run with::

    python tests/test_gamspy.py

.. note::
    By default, only unit tests are running. To enable integration tests, --integration argument should be provided.