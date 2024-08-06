.. _miro:

*********
GAMS MIRO
*********

`GAMS MIRO <https://gams.com/sales/miro_facts/>`_ (Model Interface with Rapid Orchestration) is a deployment 
environment that enables you to turn your GAMSPy model into fully-fledged end-user applications that are easy 
to distribute. You can interact with the underlying GAMSPy model, quickly create scenarios, compare results 
and visualize your data with a variety of graphical output options. 

This section provides a brief overview of the relevant commands for using GAMS MIRO with GAMSPy.  
Please refer to the `GAMS MIRO documentation <https://gams.com/miro/>`_ for more information and insights.

GAMSPy - GAMS MIRO Integration
==============================

All you need to do to use your GAMSPy models in GAMS MIRO is to annotate your MIRO input and output symbols 
that should be visible in the application. For example, the following code snippet declares symbol `d` as a 
MIRO input and symbol `x` as a MIRO output: ::

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

This initializes the default values for your GAMS MIRO app, creates the necessary data contract and spawns 
the application. To run the MIRO `Configuration mode <https://www.gams.com/miro/customize.html>`_, add the 
`--mode=config` argument: ::

    gamspy run miro --mode="config" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

To `deploy <https://www.gams.com/miro/deployment.html>`_ a GAMSPy MIRO app (create a `.miroapp` file), run 
with `--mode=deploy`: ::

    gamspy run miro --mode="deploy" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

The MIRO installation path can also be set as an environment variable with the name "MIRO_PATH" (e.g. in .bashrc), 
so that it does not have to be specified for each run. ::

    gamspy run miro --model <path_to_your_model>

This command attempts to retrieve the path to the MIRO installation from the "MIRO_PATH" environment variable. 


