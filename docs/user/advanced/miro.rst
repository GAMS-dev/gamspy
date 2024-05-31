.. _miro:

*********
GAMS MIRO
*********

With `GAMS MIRO <https://gams.com/sales/miro_facts/>`_ (Model Interface with Rapid Orchestration), you can interact 
with the underlying GAMSPy model, quickly create scenarios, compare results and visualize your data with a variety 
of graphical output options. Please refer to the `GAMS MIRO documentation <https://gams.com/miro/>`_ for more 
information and insights.

GAMSPy - GAMS MIRO Integration
==============================

All you need to do to use your GAMSPy models in GAMS MIRO is to annotate your MIRO input and output symbols. 
For example, the following code snippet declares symbol `d` as a MIRO input and symbol `x` as a MIRO output: ::

    ...
    ...
    data preparation
    ...
    ...
    
    a = Parameter(m, name="a", domain=[i], records=capacities)
    b = Parameter(m, name="b", domain=[j], records=demands)
    d = Parameter(m, name="d", domain=[i, j], records=distances, is_miro_input=True)

    x = Variable(m, name="x", domain=[i, j], type="Positive", is_miro_output=True)
    ...
    ...
    model.solve()

After you mark your miro symbols with `is_miro_input` and `is_miro_output`, you can run MIRO with the following GAMSPy
command-line utility: ::

    gamspy run miro --path <path_to_your_MIRO_installation> --model <path_to_your_model>

This initializes the default values for your GAMS MIRO app and creates the necessary data contract. Then, it spawns 
a GAMS MIRO app with base mode by default. To run the MIRO configuration mode instead, add the `--mode=config` argument: ::

    gamspy run miro --mode="config" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

To deploy a GAMSPy MIRO app (create a `.miroapp` file), run with `--mode=deploy`: ::

    gamspy run miro --mode="deploy" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

The MIRO installation path can also be set as an environment variable with the name "MIRO_PATH" (e.g. in .bashrc), so that it does not have to be specified for each run. ::

    gamspy run miro --model <path_to_your_model>

This command attempts to retrieve the path to the MIRO installation from the "MIRO_PATH" environment variable. 