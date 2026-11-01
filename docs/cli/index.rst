:html_theme.sidebar_secondary.remove:

.. _cli:

.. meta::
   :description: Reference for the GAMSPy command-line interface (CLI) used to install solvers and licenses, manage components, and run GAMSPy from the terminal.
   :keywords: GAMSPy, gamspy, GAMS, gams, CLI, command-line interface, install solver, install license, gamspy install

gamspy
======

GAMSPy comes with a command-line interface (CLI) to allow users to 
easily install solvers, licenses and much more. Autocompletion can be 
installed for the current shell with `--install-completion`.

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - -\-help 
     - -h
     - 
     - Shows the help message
   * - -\-version 
     - -v
     - 
     - Shows the version of GAMSPy, GAMS and gamspy_base
   * - -\-install-completion 
     - 
     - 
     - Install completion for the current shell.
   * - -\-show-completion 
     - 
     - 
     - Show completion for the current shell, to copy it or customize the installation.

Example
-------

Show help message::

    $ gamspy --help
    Usage: gamspy [OPTIONS] COMMAND [ARGS]...

    GAMSPy CLI

    Options:
      -h, --help     Show this message and exit.
      -v, --version  Shows the version of GAMSPy, GAMS and gamspy_base

    Commands:
      gdx       To dump and compare GDX files.
      install   To install licenses and solvers.
      list      To list solvers.
      probe     To probe node information for license retrieval.
      retrieve  To retrieve a license with another node's information.
      run       To run your model with GAMS MIRO.
      show      To show your license and gamspy_base directory.
      uninstall To uninstall licenses and solvers.

Show version information::

    $ gamspy --version
    GAMSPy version: 1.13.0
    GAMS version: 50.2.0
    gamspy_base version: 50.2.0

Using the CLI from Python
-------------------------

The commands are also available as Python functions in :mod:`gamspy.cli`. Instead of printing
their results and exiting, the functions return the results and raise
:class:`CLIError <gamspy.exceptions.CLIError>` on failure::

    from gamspy import cli
    from gamspy.exceptions import CLIError

    cli.install_solver(["highs", "scip"])         # gamspy install solver highs scip
    licensed = cli.show_license().licensed_solvers  # gamspy show license
    installed = [s.name for s in cli.list_solvers()]  # gamspy list solvers

    try:
        cli.install_license("<access_code>")       # gamspy install license <access_code>
    except CLIError as e:
        print(e.message)

List of Commands
----------------

.. toctree::
    :maxdepth: 1

    gdx
    install
    list
    mps2gms
    probe
    retrieve
    run
    show
    uninstall