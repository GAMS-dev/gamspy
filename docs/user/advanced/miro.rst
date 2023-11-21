.. _miro:

*****************
GAMS MIRO Support
*****************

`GAMS MIRO <https://gams.com/sales/miro_facts/>`_ (Model Interface with Rapid Orchestration) is a solution that 
makes it easy to turn your GAMS models into interactive end user applications. With GAMS MIRO, you can render 
GAMSPy Symbols with configurable standard graph types (Bar Charts, Scatter Plots, Pie Charts, etc), and save and 
retrieve visualizations as views.

GAMSPy - GAMS MIRO Integration
==============================

All you need to do to use your GAMSPy models in GAMS MIRO is to annotate your miro input and miro output symbols. 
For example, the following code snippet declares symbol `d` as a miro input and symbol `x` as a miro output: ::

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

After you mark your miro symbols with is_miro_input and is_miro_output, you can run MIRO with the following GAMSPy
commandline utility: ::

    gamspy run miro --path <path_to_MIRO_APPIMAGE> --model <path_to_your_model>

This initializes the default values for your GAMS MIRO app and creates the necessary data contract. Then, it spawns 
a GAMS MIRO app with base mode by default. If you want to run GAMS MIRO with another mode (e.g. config), you
can specify it with --mode argument: ::

    gamspy run miro --mode="config" --path <path_to_MIRO_APPIMAGE> --model <path_to_your_model>
