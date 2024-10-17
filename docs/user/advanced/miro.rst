.. _miro:

*********
GAMS MIRO
*********

`GAMS MIRO <https://gams.com/sales/miro_facts/>`_ (Model Interface with Rapid Orchestration) is a deployment 
environment that enables you to turn your GAMSPy models into fully-fledged end-user applications that are easy 
to distribute. You can interact with the underlying GAMSPy model, quickly create scenarios, compare results 
and visualize your data with a variety of graphical output options. 

This section provides a brief overview of the relevant commands for using MIRO with GAMSPy.  
Please refer to the `MIRO documentation <https://gams.com/miro/>`_ for more information and insights.

GAMSPy - MIRO Integration
=========================

All you need to do to use your GAMSPy models in MIRO is to annotate your MIRO input and output symbols 
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

After you mark your MIRO symbols with `is_miro_input` and `is_miro_output`, you can run MIRO with the following GAMSPy
command-line utility: ::

    gamspy run miro --path <path_to_your_MIRO_installation> --model <path_to_your_model>

This initializes the default values for your MIRO app, creates the necessary data contract and spawns 
the application. To run the MIRO `Configuration mode <https://www.gams.com/miro/customize.html>`_, add the 
`--mode=config` argument: ::

    gamspy run miro --mode="config" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

To `deploy <https://www.gams.com/miro/deployment.html>`_ a GAMSPy MIRO app (create a `.miroapp` file), run 
with `--mode=deploy`: ::

    gamspy run miro --mode="deploy" --path <path_to_your_MIRO_installation> --model <path_to_your_model>

The MIRO installation path can also be set as an environment variable with the name "MIRO_PATH" (e.g. in .bashrc), 
so that it does not have to be specified for each run. ::

    gamspy run miro --model <path_to_your_model>

This command attempts to retrieve the path to the MIRO installation from the "MIRO_PATH" environment variable. You can also skip defining the ``--path`` argument 
in case your MIRO installation is in one of the standard paths: 

- macOS: /Applications/GAMS MIRO.app/Contents/MacOS/GAMS MIRO or ~/Applications/GAMS MIRO.app/Contents/MacOS/GAMS MIRO
- Windows: C:\\Program Files\\GAMS MIRO\\GAMS MIRO.exe or C:\\Users\\<username>\\AppData\\Local\\Programs\\GAMS MIRO\\GAMS MIRO.exe

When running a GAMSPy job from MIRO, you may not want to perform certain expensive operations, such as loading MIRO input data from an Excel workbook, as this data comes from MIRO.
In that case, one can conditionally load the data by using the ``in_miro`` attribute of `Container`. For example: ::
    
    import pandas as pd
    from gamspy import Container, Parameter
    
    m = Container()

    f = Parameter(
        m,
        name="f",
        description="supply of commodity at plant i (in cases)",
        records=pd.read_excel(my_large_xlsx) if not m.in_miro else None,
        is_miro_input=True,
    )

The script above would only load the excel file if the GAMSPy script is not run with MIRO. For large data files 
this option would improve performance.