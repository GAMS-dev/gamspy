.. _installation:

===================
Installation
===================

Basic Installation
------------------

You can run the following command to install GamsPy from PyPI:

    $ pip install gamspy

Building from source
--------------------

If you are a MacOS or Linux user familiar with using the command line, 
you can continue with building GamsPy locally by following the instructions below.

Prerequisites
-------------

Building GamsPy requires the following software installed:

1) Python 3.8.x or newer

2) The GamsPy source code

    $ git clone <prefix>/gamspy.git

Basic Build and Installation
----------------------------

To install , run::

    $ python setup.py sdist bdist_wheel 
    $ pip install gamspy[test,dev] --find-links dist/

Testing
-------

Run tests with::

    $ python tests/test_gamspy.py