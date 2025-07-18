GAMSPy 1.13.0 (2025-07-15)
==========================

New features
------------
- #616: Implement container.writeSolverOptions to write solver option files to the working directory.
- #654: Implemented `TorchSequential` convenience formulation for embedding Sequential layers into GAMSPy.

Improvements in existing functionality
--------------------------------------
- #658: Add a gamspy option to disable solver validation. Useful for solvers that are added via gamsconfig.yaml.
- #659: Allow server and port specification for on-prem license servers.
- #660: Add `DROP_DOMAIN_VIOLATIONS` option.

Bug fixes
---------
- #657: Unbounded input in the RegressionTree caused the value of M to become `infinity`. To prevent this, limit M to 1e10.
- #663: Fix duplicate domain name issue in the MIRO contract.
  Fix symbol declaration without records for miro apps.
- #665: Allow PathLike objects for loadpoint option.
- #666: Fix set attributes records call.
- #667: Fix the bug in expert sync mode due to missing attribute.

Improved documentation
----------------------
- #654: Added docs for `TorchSequential` formulation.

GAMSPy 1.12.1 (2025-07-07)
==========================

Improvements in existing functionality
--------------------------------------
- #658: Add a gamspy option to disable solver validation. Useful for solvers that are added via gamsconfig.yaml.
- #659: Allow server and port specification for on-prem license servers.
- #660: Add `DROP_DOMAIN_VIOLATIONS` option.

Bug fixes
---------
- #657: Unbounded input in the RegressionTree caused the value of M to become `infinity`. To prevent this, limit M to 1e10.
- #663: Fix duplicate domain name issue in the MIRO contract.
  Fix symbol declaration without records for miro apps.

GAMSPy 1.12.0 (2025-06-21)
==========================

New features
------------
- #579: Add Decision Tree formulation.
- #637: Add Conv1d formulation.

Improvements in existing functionality
--------------------------------------
- #561: Allow args and kwargs for gamspy scripts for miro applications.
- #645: Accept os.PathLike objects as load from argument of Container.
- #646: Allow providing an explicit path to the license file.
- #648: Improve static typing.

Improved documentation
----------------------
- #579: Add classic machine learning formulations to the documentation.
- #637: Add docs for Conv1d formulation.

GAMSPy 1.11.1 (2025-06-03)
==========================

Improvements in existing functionality
--------------------------------------
- #611: Allow users to create GAMSPy options from a dictionary of GAMS options.
- #640: Add "append_to_log_file" option to allow appending to the log file.

Bug fixes
---------
- #641: Fix the overload of the unary magic function.
- #643: Escape underscores in latex representation to support symbol names with underscores.

GAMSPy 1.11.0 (2025-05-16)
==========================

New features
------------
- #607: Allow evaluation of expression on the fly without requiring an explicit assignment statement to a parameter.

Bug fixes
---------
- #608: Fix literal bug in latex representation of implicit symbols.
- #633: Incrementally build model declaration to avoid input line length limitation (80000 characters).
- #638: Fix the bug that occurs when "gamspy probe -h" runs.

Improved documentation
----------------------
- #614: Add developer guide to the documentation.

CI/CD changes
-------------
- #631: Add tests for Linux arm64. Add a new marker called "requires_license" to separate tests that require a license to run.

GAMSPy 1.10.0 (2025-04-30)
==========================

New features
------------
- #599: Allow renaming on ``container.loadRecordsFromGdx`` function call.
- #601: Allow users to disable all validation via ``gp.set_options({"VALIDATION": 0})``.

Improvements in existing functionality
--------------------------------------
- #594: Add GAMSPyJacobian file format for the convert function.
- #612: Raise a validation error in case an automatically named symbol is used in an equation of an EMP model. Use base64 auto-generated names instead of plain uuid.uuid4 names.
- #613: Improve the error message of undefined scalar equations.
- #615: Remove duplicate conopt entries in gamspy list solvers cli command and add utils.getInstallableSolvers function.
- #617: Cast the type of objective value, num equations, num variables, and solver time in the summary dataframe.
- #623: Improve the error message in case the user does not have an internet connection or the PyPI server are down.
- #624: Allow .records call on implicit variables and equations.

Bug fixes
---------
- #625: Overload __eq__ and __ne__ magic functions of the Number class to ensure the order is correct in expressions.
- #626: Fix the bug in the filtering of a single record in non-level attributes of a variable (lo, up, marginal, scale).
- #629: Allow record filtering over eq.range, eq.slacklo, eq.slackup, eq.slack, and eq.infeas attributes.

Improved documentation
----------------------
- #602: Use towncrier to automate changelog creation and avoid marge conflicts in the changelog file.

GAMSPy 1.9.0
============
- General
  - Validate solver options for most of the solvers. It can be disable through VALIDATE_SOLVER_OPTIONS option.
  - get the value of objective estimation and the number of nodes used after frozen solves
  - Add description argument for model objects.
  - Make GAMSPy symbols non-iterable.
  - Inherit output argument from the container in solve function if the output argument is not specified.
  - Start the renaming process (deprecation) of model_instance_options to freeze_options. GAMSPy 1.9.0 will throw a warning. GAMSPy 1.10.0 will throw an exception, and GAMSPy 1.11.0 will remove model_instance_options altogether.
  - Fix sense=feasibility bug of frozen models. 
  - Rename ModelInstanceOptions to FreezeOptions and add a warning for the usage of ModelInstanceOptions.
  - Add model.convert function to allow converting a GAMSPy model instance to different file formats.
  - Fix sense=feasibility bug of frozen models.
  - Fix static code analysis errors.
  - Do not validate equation definitions in case the container was restarted from a save file (.g00 file).
  - Propagate the output option of the container to `model.freeze`.
  - Raise warning in case the frozen solve is interrupted.
  - Improve the performance of symbol declarations without any records and declaration of 0 dimensional symbols with records.
- Documentation
  - Add additional instructions to deploy a GAMSPy/GAMS MIRO model.
  - Fix name mismatch between the argument name and the docstring of loadRecordsFromGdx function.
- Testing
  - Run all pre-commit hooks instead of running selectively.

GAMSPy 1.8.0
============
- General
  - Improve the performance of frozen solves.
  - Add support for new equation, variable matching syntax for MCP models.
  - Ignore empty and newlines in the existing solvers file.
  - Use finalizers instead of __del__.
  - Cache solver capabilities, default solvers and installed solvers to speed up solver validation.
  - Fix the bug in the case of multiple frozen models in one container.
  - Perform pip audit check in the pipeline instead of pre-commit.
  - Upgrade gamspy_base and gamsapi dependencies.
- Documentation
  - Add `Examples` section under `Machine Learning` documentation.
  - Add a Thermal Reformer example demonstrating neural network surrogate modeling.
- Testing
  - Fix the issue of mac jobs deleting each others environments.

GAMSPy 1.7.0
============
- General
  - Allow container serialization/deserialization.
  - Support an alternative syntax for operations. For example, x.sum() is equivalent to Sum(x.domain, x[x.domain]).
  - Fix a bug when starting from a GAMS restart file.
  - Allow propagating bounds of the output in `Conv2D` class.
  - Introduce `name_prefix` option to NN formulations for ease of debugging.
- Documentation
  - Add a section in FAQ about the compatibiltiy issues of the Python interpreter from the Microsoft Store.
  - Fix minor issue in embedding Neural Network documentation.
- Testing
  - Enforce the order of tests. Run unit tests first, and model library tests last.
  - Use spawn method for multiprocessing to avoid possible deadlocks with fork method.

GAMSPy 1.6.0
============
- General
  - Upgrade pre-commit dependencies.
  - Enhance bound propagation for `AvgPool2d` class.
  - Allow adding debug options to Options objects.
  - Allow starting from a GAMS restart file.
  - Allow registering metadata to symbols via `symbol._metadata` attribute.
  - Fix solver option format of HIGHS, SHOT, SOPLEX and SCIP.
  - Allow dumping gams state on `.toGams` call.
  - Allow indexing into symbols with integers.
  - Add `bypass_solver`, `cutoff`, and `default_point` options.
  - Replace conda, pip and virtualenv with uv in ci pipelines.
  - Add --use-uv option to allow solver downloads with uv.
  - Provide ssl context explicitly for NEOS backend.
  - Add configurable options via set_options and get_option.
  - Fix bug in an edge case of the vector-matrix multiplication.
- Testing
  - Add an lp and a qcp benchmark for performance comparison.
- Documentation
  - Add CNNs to embedding Neural Network documentation.

GAMSPy 1.5.1
============
- General
  - Fix the bugs in dynamic set assignments.
  - Rewrite parts of GAMS Control API.
  - Fix debugging level bug of NEOS backend.
  - Fix license issue of big models that are solved with frozen solve.
  - Allow loadRecordsFromGdx to domain forward.
  - Enhance bound propagation for `MaxPool2d` and `MinPool2d` classes.
- Testing
  - Add bound propagation tests for `MaxPool2d` and `MinPool2d` classes.
- Documentation
  - Update embedding Neural Network documentation.

GAMSPy 1.5.0
============
- General
  - Fix implicit parameter validation bug.
  - Migrate GAMSPy CLI to Typer.
  - Threads can now create a container since we register the signal only to the main thread.
  - Fix solver options bug in frozen solve.
  - Synchronize after read.
  - Upgrade gamspy_base and gamsapi dependencies.
  - Add `--checkout-duration` and `--renew` options to `gamspy install license`.
- Testing
  - Lower the number of dices in the interrupt test and put a time limit to the solve.
  - Add tests for piecewise linear functions.
- Documentation
  - Install dependencies in the first cell of the example transportation notebook.
  - Add Formulations page to list piecewise linear functions and nn formulations.

GAMSPy 1.4.0
============
- General
  - Resolve static code analysis issues to improve code quality.
  - Return the value as a float if the given domain sets are all literals.
  - Add an automation script to update pyproject.toml, switcher, version test, and the release notes.
  - Allow propagating bounds of the output in the Linear class.
  - Allow GAMS to find the available port and connect to it.
  - Upgrade gamspy_base and gamsapi dependencies.
- Testing
  - Set COVERAGE_CORE to sysmon to make use of the new sys.monitoring package in Python.
- Documentation
  - Add an example demonstrating how to solve the Minimum Cost Multi-Commodity Flow Problem using Column Generation in GAMSPy.
  - Remove non-negative variable type from the docs.
  - Add plausible.js for analytics.
  - Minor update in embedding nn documentation.
  - Add descriptions and example code to formulations documentation.


GAMSPy 1.3.1
============
- General
  - Fix the bug in equality type traversal. Use post-traversal instead of in-order traversal.

GAMSPy 1.3.0
============
- General
  - Change the way to show limited variables in latex file.
  - Overload __rpower__ for operables.
  - Support __neg__ overload for Card and Ord operations.
  - Fix the bug in new lag/lead syntax.
  - Add a verification step for working directory path length.
  - Add `map_value` function to the math library.
  - Allow conditioning on conditions.
  - Upgrade gamspy_base and gamsapi dependencies. 
- Documentation
  - Add a section for limited variables. 
  - Add an example that shows how to read from another Container.

GAMSPy 1.2.0
============
- General
  - Fix non-zero return code issue in case there is an error in the script. In case the return code is non-zero, GAMSPy will not launch GAMS MIRO.
  - Fix the behaviour of CTRL+C. 
  - Allow alternative `set +/- n` syntax for lead and lag operations. 
  - Upgrade gamspy_base and gamsapi dependencies.
  - Expose the filename and the line number of the solve to the listing file.
  - Improve the performance of `load_from` argument of Container.
- Testing
  - Add a new performance test which compares the performance of GAMS Transfer read and GAMSPy read.
- Documentation
  - Add a favicon.

GAMSPy 1.1.0
============
- General
  - Allow printing the records of variable/equation attributes with a print(variable.attribute[domain].records) syntax.
  - Allow printing the records of a subset of a parameter with print(parameter['literal'].records) syntax.
  - Allow printing the records of a subset of a set with print(set['literal'].records) syntax.
  - Update variable/equation attribute domains on addGamsCode call.
  - Show log file instead of listing file on solve statements with NEOS backend.
  - Add Linear layer formulation
  - Fix minor bug of domain conflict in batched matrix multiplication
  - Improve the error messages of the thrown exceptions in case the user provide a model option at Container creation time.
  - Do not allow models with the same name to override each other.
- Testing
  - Fix race conditions in the pipeline.
  - Remove redundant setRecords in gapmin.py example.
  - Add sq.py model to the test model suite.
  - Update hansmge model.
  - Fix lower bound in reshop model.
  - Add tests for the Linear layer
  - Add a script to measure the overhead of GAMSPy and Python in general for each model in the model library.
- Documentation
  - Add documentation for the Linear layer

GAMSPy 1.0.4
============
- General
  - Do not create a GDX file when it's not necessary. 
  - Do not carry solver options from the previous solve to the new solve.
  - Fix toGams bug of MathOp symbols.
  - Use symbol< syntax of GAMS to handle domain forwarding.
  - Add "same" and "valid" options for Conv2d padding.
  - Update dependencies. gamspy_base -> 48.1.1 and gamsapi -> 48.1.0.
  - Make minimum supported Python version 3.9 and add support for Python 3.13.
- Documentation
  - Fix documented type of model.solve_status.
  - Add num_equations attribute to the model page of user guide.
  - Add synchronization docs to reference api.
- Testing
  - Add one to one comparison tests with reference files in toGams tests.
  - Add tests for "same" and "valid" padding options of Conv2d.

GAMSPy 1.0.3
============
- General
  - Fix solver installation bug in case of a solver installation before the license installation.
  - Fix the validation bug on multiple operations in a row.
  - Fix set attribute comparison bug.
- Testing
  - Remove leftover files after running all tests.

GAMSPy 1.0.2
============
- General
  - Validate whether the solver is installed only for local backend.
  - Change the default value of sense to Sense.FEASIBILITY.
  - Support output in Container constructor.
  - Fix debugging_level bug.
  - Add additional checks for the validity of the license.
  - Allow generateGamsString function only if the debugging level is set to "keep".
  - Fix socket communication issue on license error.
  - Distinguish GamspyException from FatalError. The user might catch GamspyException and continue but FatalError should never be caught.
  - Fix singleton assignment bug.
  - Allow an alternative syntax for variable/equation attributes (e.g. b[t].stage = 30).
  - Add support for MaxPool2d/MinPool2d/AvgPool2d.
  - Add support for flatten_dims for flattening n domains into 1 domain.
  - Show class members groupwise in the table of contents (first methods, then properties). 
  - Use the new license server endpoint to verify the license type.
  - Don't do extra unnecessary GAMSPy to GAMS synch after addGamsCode.
  - Fix incorrect domain information of symbols created by addGamsCode 
  - Fix network license issue on NEOS Server.
  - Replace non-utf8 bytes of stdout.
- Testing
  - Remove license uninstall test to avoid crashing parallel tests on the same machine.
  - Add tests for the generated solve strings for different type of problems.
  - Add a test for Container output argument.
  - Add tests for debugging_level.
  - Add tests to verify the validity of the license.
  - Add memory check script for the performance CI step.
  - Add tests for the alternative syntax for variable/equation attributes.
  - Add tests for pooling layers and flatten_dims
- Documentation
  - Fix broken links in the documentation.
  - Add a ci step to check doc links.
  - Improve the wording of debugging document.
  - Add pooling and flatten_dims docs.

GAMSPy 1.0.1
============
- General
  - Fix frozen solve with non-scalar symbols.
  - Fix the definition update problem while redefining an equation with definition argument.
  - Introduce default directories to keep license information on upgrade.
  - Add --existing-solvers and --install-all-solvers options for gamspy install solver.
  - Add --uninstall-all-solvers option for gamspy uninstall solver.
  - Show license path on gamspy show license command.
  - Simplify the implementation of the copy container operation.
  - Add Conv2d formulation for convenience
  - Map GAMSPy problem types to NEOS problem types before sending the job.
  - Upgrade gamspy_base and gamsapi versions to 47.6.0. 
- Testing
  - Add test for the frozen solve with non-scalar symbols.
  - Add a test to verify the behaviour of equation redefinition with definition argument.
  - Test the usage of a license that is in one of the default paths.
  - Fix the issue related to reading equation records from a gdx file.
  - Add tests to verify the records after reading them from a gdx file.
  - Add tests for installing/uninstalling solvers.
  - Add tests to verify correctness of Conv2d formulation
  - Add a test to verify GAMSPy -> NEOS mapping.
  - Add an execution error test.
- Documentation
  - Update the documentation of install/uninstall command line arguments.
  - Add a section for NN formulations

GAMSPy 1.0.0
============
- General
  - Fix starting from a loadpoint for GAMS Engine backend.
  - Fix solver options issue for GAMS Engine backend.
  - Fix solver options issue for NEOS backend.
  - Support external equation for GAMS Engine backend.
  - Change the behaviour of expert synch mode.
  - Update quick start guide with latex to pdf output.
  - Fix quote issue in paths.
  - Activation functions now return added equations as well.
  - skip_intrinsic option added for log_softmax.
  - Allow installing/uninstalling multiple solvers at once.
  - Make miro_protect an option.
  - Show a better help message on gamspy -h command.
  - Fix missing links in api reference.
  - Set default problem type as MIP instead of LP.
  - Allow UniverseAlias in assignments.
  - Add performance ci step to check model generation time difference.
  - Update gamspy_base and gamsapi to 47.5.0.
- Documentation
  - Add a warning about the manipulation of records via .records. 
  - Fix model attribute return type.
- Testing
  - Add sat problem to the example models.

GAMSPy 0.14.7
=============
- General
  - Include variable infeasibilities in model.computeInfeasibilities().
  - Remove cone equation type.
  - Fix empty space issue in paths.
- Documentation
  - Add gamspy probe and gamspy retrieve to the cli reference page.
  - Fix typo in miro docs.

GAMSPy 0.14.6
=============
- General
  - Fix GAMS Engine get_logs return values according to the status code.
  - Allow explicit port definition via environment variable to communicate with GAMS. 
  - Replace GamsWorkspace with GAMSPy workspace implementation.
  - Remove unnecessary validation for system_directory.
  - Better formatting for gamspy list solvers and gamspy list solvers -a.
  - Change the structure installing licenses on offline machines.
  - Fix UniverseAlias bug.
  - Check standard locations for GAMS MIRO.
  - Simplify toLatex output.
  - Make name optional for addX syntax of adding symbols.
  - Add __mod__ overload for all operables.
  - Fix domain forwarding issue when trying to forward records to the same set.
  - Do not convert eps to zero by default.
  - Add Sand and Sor operations.
  - Ensure that external equations contain == operation.
- Testing
  - Use the Container that is created in the setup phase instead of creating a new one.
  - Remove unnecessary init files in tests.
  - Add a test for invalid port.
  - Explicitly close the Container for jobs executed by ProcessPoolExecutor.
  - Add a test for long running jobs with network license.
  - Add tests for gamspy probe and gamspy retrieve license.
  - Add test to use UniverseAlias object as domain.
  - Add tests to verify that symbol creation with no name is possible.
- Documentation
  - Add what is gamspy page to docs.
  - Update indexing docs.
  - Add a link to model library on the landing page.
  - Encourage the use of the Discourse platform instead of sending direct emails to gamspy@gams.com. 
  - Add instructions on how to install a license on an offline machine.
  - Update what is gamspy page model example.
  - Change the order of symbol declaration and data specification in the quick start guide.
  - Add equation listing, variable listing, and interoperabiltiy sections to quick start guide.
  - Add gamspy.exceptions to the api reference.
  - Change the order of indexing, lag-lead operations, ord-card operations and number.
  - Add gamspy.NeosClient to the api reference.
  - Add model attributes to docstring.

GAMSPy 0.14.5
=============
- General
  - Retry login with exponential backoff in GAMS Engine backend.
  - Allow to set all model attributes that can be set before solve in GAMS.
  - Fix equation listing, variable listing parsing when listing file is specified.
- Testing
  - Use contextmanager to create atomic conda environments.
  - Add tests for model attribute options.
- Documentation
  - Fix links in the api reference.
  - Add an example that shows how to embed NN to an optimization problem.

GAMSPy 0.14.4
=============
- General
  - Add container.in_miro flag to selectively load data.
  - Parse error message after verifying the return code for engine backend.
  - Fix the behaviour of Model if it's declared twice with objective function.
  - Update the error message of license error.
  - Fix output stream validation.
  - Fix exception on solve in case listing file is specified.
  - Add external equations support.
  - Do not raise exception in case GAMS Engine returns 308 on get_logs call.
- Testing
  - Add test for container.in_miro flag.
  - Add tests to simulate Jupyter Notebook behaviour.
  - Remove system_directory for tests.
  - Add a test which specifies the listing file and fails because the license does not allow to run the model.
  - Add tests for external equations support.
  - Add traffic model to the model library.
- Documentation
  - Document in_miro flag.
  - Add docstring for setBaseEqual.
  - Add section "External Equations" under Advanced documentation.
  - Add section "Extrinsic Functions" under Advanced documentation.

GAMSPy 0.14.3
=============
- General
  - Add getEquationListing function to be able to inspect generated equations.
  - Add infeasibility threshold filter for equation listings.
  - Add getVariableListing function to be able to inspect generated variables.
- Testing
  - Add tests for getEquationListing function.
  - Add tests for getVariableListing function.
  - Test infeasibility threshold.
- Documentation
  - Add docs for getEquationListing.
  - Add docs for getVariableListing.

GAMSPy 0.14.2
=============
- General
  - Add generate_name_dict option.
  - Disable solution report by default.
  - Fix the order of equations in toGams utility.
  - Allow options in toGams.
  - Add loadpoint option to start from a solution.
  - Upgrade gamspy_base and gamsapi to 47.4.0.

GAMSPy 0.14.1
=============
- General
  - Add SOS1 ReLU implementation.
  - Add __repr__ to all GAMSPy language constructs for better debugging.
  - Give a warning in case the domain is not initialized by the time there is an attribute assigment.
  - Allow indexing on alias symbols.
  - Add reference_file option.
  - Add selective loading for solve statements.
  - Change default port to communicate with license server to 443.
  - Fix installing licenses from a path.
- Documentation
  - Add API docs for SOS1 ReLU implementation.
  - Explain the working directory - debugging level relationship.
- Testing
  - Add tests for SOS1 ReLU implementation.
  - Shorten attribute assignments in model library (variable.l[...] = ... -> variable.l = ...).
  - Add tests for indexing on alias symbols.
  - Test selective loading for solve statements.
  - Add new install license tests.
  - Add a new model (coex) to the model library.


GAMSPy 0.14.0
=============
- General
  - Introduce matrix multiplication operator `@`.
  - Add most common activation functions for machine learning.
  - Improve domain checking.
  - Write division with frac in toLatex function.
  - Allow specifying port for the communication with GAMS license server with --port argument of GAMSPy CLI.
- Documentation
  - Add GAMSPy and Machine Learning section.
  - Add ML examples.
  - Give more information about the restrictions of frozen solve.
- Testing
  - Add tests for different cases of matrix multiplication.
  - Add tests for activation functions.
  - Add tests for domain checking.
  - Shorten refrigerator example model by folding repetitive code into loops.


GAMSPy 0.13.7
=============
- General
  - Support .where syntax for Card and Ord.
  - Return condition on where operations on the right instead of expression.
  - Support custom streams for output redirection.
  - Catch set is already under control errors early.
- Documentation
  - Fix docstring of the Card operation.
  - Add warning about non-professional licenses in addGamsCode docstring.
  - Add an example to show how to redirect output to a custom stream.
- Testing
  - Add tests for .where syntax for Card and Ord.
  - Add tests to catch set is already under control errors.
  - Add a test which redirects output to a custom stream.

GAMSPy 0.13.6
=============
- General
  - Make all file read and writes with utf-8 encoding.
  - Fix model instance record columns.
  - Allow all iterables for equations argument of model.
  - Fix the bug in socket connection messages.
- Testing
  - Add a test to verify the columns of symbols in model instance solves.
  - Test set difference for model equations argument.

GAMSPy 0.13.5
=============
- General
  - Make trace file name dynamic to avoid race condition on parallel runs.
  - Fix log options for GAMS Engine backend.
  - Initial support for GAMSPy to Latex.
  - Generate solver options file under container working directory instead of current directory.
  - Fix implicit set issues for toGams function.
- Documentation
  - Add links to the api reference for symbols and functions mentioned in the documentation.
  - Minor documentation corrections.
- Testing
  - Logout from GAMS Engine only on Python 3.12 to avoid unauthorized calls on parallel jobs.
  - Add tests to verify the behaviour of different logoption values.
  - Add tests for GAMSPy to Latex.

GAMSPy 0.13.4
=============
- General
  - Fix hanging issue on Windows for GAMS Engine backend.
  - Refactor toGams converter.
  - Fix solver options file path bug.
- Testing
  - Add more tests for GAMS MIRO.

GAMSPy 0.13.3
=============
- General
  - Change default solvers to 'CONOPT', 'CONVERT', 'CPLEX', 'GUSS', 'IPOPT', 'IPOPTH', 'KESTREL', 'NLPEC', 'PATH', and 'SHOT'
  - Fix the version of gamspy_base when "gamspy update" command is being executed.
  - Fix the order issue for Alias in toGams function.
  - Add exponential backoff for GAMS Engine logout api.
  - Add symbol validation for Ord operation.
- Testing
  - Update model library tests according to the new default solvers.
  - Add a test to verify that modifiable symbols cannot be in conditions for model instance runs.
  - Add new tests for symbol validation.

GAMSPy 0.13.2
=============
- General
  - Set the records of objective value in model instance solves. 
  - Allow using an environment variable to set the GAMS system directory (given environment variable will override the system directory even if the user provides a system directory argument to Container).
  - Use gdxSymbols commandline option instead of manually marking symbols dirty.
  - Add memory_tick_interval, monitor_process_tree_memory, and profile_file options.
  - Change the way to generate GAMS model from a GAMSPy model.
  - Remove import_symbols argument for addGamsCode since it is not needed anymore.
- Documentation
  - Redirect model library page to gamspy-examples Github repo.
  - Update toGams docs.
  - Update doctest of addGamsCode.
- Testing
  - Add model instance tests that check the objective value.
  - Update system directory test to adjust to the environment variable support.
  - Add tests for profiling options.

GAMSPy 0.13.1
=============
- General
  - Support output redirection for NEOS backend.
  - Support GAMSPy to GAMS automatic conversion.
  - Add support for old way of installing a license. 
- Documentation
  - Update model documentation to show how to redirect NEOS output to a file.
  - Add examples to all public functions in API Reference.
- Testing
  - Add a new model (knapsack) to the model library.

GAMSPy 0.13.0
=============
- General
  - Communicate with GAMS executable via socket instead of spawning a new job everytime.
- Documentation
  - Adjust debugging page according to the new .gms generation rules.
  - Update installation page to adjust to the new licensing scheme.
- Testing
  - Add new tests to verify correct license installation and listing solvers.

GAMSPy 0.12.7
=============
- General
  - Fix equation/variable listing bug.
  - Exclude autogen statements in generateGamsString raw.
  - Upgrade gamspy_base and gamsapi versions to 47.1.0.
  - Fix parameter equality bug in equations.
  - Set upper bound of numpy version below 2 until gamsapi supports it.
- Documentation
  - Fix the alignment of code section in debugging page.
- Testing
  - Add test to verify the correctness of parameter equality in equations.

GAMSPy 0.12.6
=============
- General
  - Do not open gdx file in case there is nothing to load.
  - Fix solver capability check bug.
  - Enable explicit expert synchronization for symbols.
  - Fix dist function in math package.
Testing
  - Adapt generateGamsString tests to new the gdx load logic. 
  - Add test for the solver capability bug.
  - Test explicit expert synchronization for symbols.

GAMSPy 0.12.5
=============
- General
  - Do not pick the default solver if the given solver is not compatible with the problem type.
  - Add extrinsic function support.
  - Expose addGamsCode to user.
  - Refactor the underlying implementation of options.
  - Show better error messages.
  - Fix number of arguments that log_gamma takes.
  - Rename getStatement to getDeclaration.
- Testing
  - Add tests for extrinsic functions.
  - Test whether the given solver is capable of solving the problem type.
  - Add an addGamsCode test for each problem type. 
  - Test Jupyter Notebooks in docs automatically.
  - update log option tests.
- Documentation
  - Remove unnecessary GTP functions from documentation
  - Add a doctest for addGamsCode.
  - Update the documentation on generating log files.

GAMSPy 0.12.4
=============
- General
  - Add checks on model name.
  - Adjust when to throw an exception and when to throw a warning for different SolveStatus values.
  - Make autogenerated model attribute symbol names independent of the model name.
  - Do not allow expressions and symbols to be used as truth values.
  - Add deprecation message for getStatement and expose getDeclaration and getDefinition.
  - Override __repr__ and __str__ of Container.
  - Synchronize gamspy_base and gamsapi versions.
- Testing
  - Test invalid model names.
  - Add tests for expressions and symbols that are used as truth values.
  - Add tests for __repr__ and __str__ of Container.

GAMSPy 0.12.3
=============
- General
  - Set log and listing file option relative to os.cwd instead of workspace.working_directory.
  - Simplify expression generation and fix incorrect expression data. 
  - Add logoption=4.
  - Add show_raw option to the generateGamsString function.
- Testing
  - Test relative path for listing file and log file creation options.
  - Update log option tests.
  - Add new tests for generateGamString.
- Documentation
  - Remove the remnants of .definition and .assignment syntax from documentation.
  - Fix the example in gamspy for gams users.
  - Add notes about the equivalent operation in GAMS to .where syntax in GAMSPy.
  - Update the documentation for debugging with generateGamsString.

GAMSPy 0.12.2
=============
- General
  - Add infeasibility_tolerance as a model attribute.
  - Make urllib3 a true dependency instead of an optional one.
  - Do not suppress compiler listing by default.
  - Improve the performance of model attribute loading.
  - Load miro input symbols once.
  - Fix license path for model instance.
- Documentation
  - Add documentation about solver specific infeasibility options.

GAMSPy 0.12.1
=============
- General
  - Fix dataframe column names of GAMS MIRO input tables.
  - Catch solve status errors and throw necessary exceptions.
  - __pow__ returns sqrt instead of rPower if the exponent is 0.5.
  - Deprecate delayed_execution mode.
  - Replace pylint, flake8 and black with ruff.
  - Implement /api/auth -> post, /api/auth/login -> post and /api/auth/logout -> post for GAMS Engine.
  - Allow dumping log file to arbitrary path.
  - Allow dumping listing file to arbitrary path.
  - Allow dumping gdx file to arbitrary path.
  - Disallow equation definitions without any equality sign.
  - Add calculate_infeasibilities function for variables, equations and models.
  - Add 'gamspy show license', and 'gamspy show base' commands.
  - Replace __del__ with atexit.register function.
- Testing
  - Replace cta PandasExcelReader and PandasExcelWriter with new ExcelReader and ExcelWriter from GAMS Connect correspondingly. 
  - Add a new model (Nurses) to the model library and the Notebook examples.
  - Add an AC optimal power flow (ACOPF) model to the model library.
  - Add a test to verify the generated string for power where the exponent is 0.5.
  - Add tests for /api/auth.
  - Add a test for creating log file with arbitrary name.
  - Add a test for creating lst file with arbitrary name.
  - Add a test for creating gdx file with arbitrary name.
  - Add tests for infeasibility calculations.
- Documentation
  - Remove FAQ about Google Colab (it is resolved) and add FAQ about Windows Defender.
  - Remove documentation for delayed execution mode.
  - Add an example for providing solver options.
  - Document CLI for gamspy show commands.

GAMSPy 0.12.0
=============
- General
  - Implement GAMS MIRO integration.
  - Update minimum gamsapi and gamspy_base version to 46.1.0.
- Testing
  - Add tests for GAMS MIRO.
- Documentation
  - Add documentation of GAMS MIRO integration.
  
GAMSPy 0.11.10
==============
- General
  - Adapt debugging level to GAMS 46 debugging levels.
  - Adapt getInstalledSolvers to renaming of SCENSOLVER
- Testing
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Update the results of model instance tests (CONOPT3 -> CONOPT4).

GAMSPy 0.11.9
=============
- General
  - Fix relative path issue of GAMS Engine backend.
  - Use $loadDC instead of $load to better catch domain violations.
  - Bypass constructor while creating a Container copy.
  - Do not execute_unload in case there is no dirty symbols to unload.
  - Update the behavior of `gamspy install/uninstall license`.
  - Implement GAMS Engine Client and consolidate NeosClient and EngineClient into one argument in solve.
  - Fix finding variables to mark in power and sameAs operations.
- Testing
  - Add test for GAMS Engine extra model files with incorrect relative path.
  - Add tests for new GAMS Engine Client.
  - Add a test to catch domain violation.
  - Remove declaration of objective variables and functions and add the equations into Python variables.
  - Add a new test to verify the license installation/uninstallation behavior.
  - Add a test to find variables in power operation.
- Documentation
  - Add a note in model documentation to warn about relative path requirement of GAMS Engine.
  - Add documentation for solving models asynchronously with GAMS Engine.
  - Modify model library table generation script to add more information and better table styling.

GAMSPy 0.11.8
=============
- General
  - Allow assigning VariableType enum or EquationType enum as an attribute after object creation for Equation and Variable.
  - Fix running twice on symbol declaration with records
  - Return better error messages for incorrectly provided solver, options, and output arguments.
  - Fix missing uels_on_axes argument in setRecords.
  - Start using pylint to improve code quality.
- Testing
  - Add tests for assigning type to Variable and Equation after creation.
  - Add models information at the top of each model's docstring.
  - Add tests for setRecords with uels on axes.
- Documentation
  - Add docs for translating GAMS Macros to GAMSPy.

GAMSPy 0.11.7
=============
- General
  - Implement GAMS MIRO integration.
  - Allow variable/equation attribute assignment without any index.
  - Run GAMS on symbol declaration and setRecords.
  - Add debugging_level argument to Container.
  - Performance improvements (~25-30%).
- Testing
  - Add tests for GAMS MIRO.
  - Test scalar variable/equation assignment without any index.
  - Test uel order.
- Documentation
  - Add documentation of GAMS MIRO integration.
  - Document assigning to scalar variable/equation.
  - Update documentation of frozen solve (model instance in GAMS). 
  - Add documentation for debugging levels.

GAMSPy 0.11.6
=============
- General
  - Support slices for indexing.
  - Fix unary operator for expressions
  - Fixes SpecialValues bug in expressions.
  - Fix the bug for nonbinding equations.
  - Fix discovery of variables in math operations.
  - Fix literal while checking for domain validation.
- Testing
  - Add tests for slicing and ellipsis.
  - Add tests for unary operator applied to expressions.
  - Add tests to verify the correctness of SpecialValues in expressions.
  - Add more tests for nonbinding equations.
- Documentation
  - Document indexing with slices and ellipsis.


GAMSPy 0.11.5
=============
- General
  - Verify dimensionality of the symbol and the given indices to provide better error messages.
  - Allow Model object to also accept tuple of equations.
  - List available and installed solvers in alphabetically sorted order.
  - Fix adding autogenerated equations twice. 
  - Generate unique names for the autogenerated variables and equations.
  - Add __str__ and __repr__ to Model.
  - Allow literals in sameAs operation.
  - Make Number operable.
  - Add more data validation functions.
  - Clear autogenerated symbols from the container if there is an exception.
  - Fix Alias bug while preparing modified symbols list.
- Testing
  - Add tests to check if incompatible dimensionality throws exception.
  - Test validation errors.
  - Allow providing system directory for the tests via environment variable.
- Documentation
  - Add documentation for `matches` argument of Model.


GAMSPy 0.11.4
=============
- General
  - Expose GTP special values via gp.SpecialValues
  - Fix NEOS bug when the previous workfile comes from another backend.
  - Optimize read function of Container by assigning the symbols' attributes directly while casting.
  - Remove autogenerated variable and equation from Container after each solve.
  - Recover dirty and modified states if the write is invoked by the user.
  - Do not expose cast_to_gamspy to user.
  - Abstract backends to allow easier extension.
  - Add compress, mode, eps_to_zero arguments to write
  - Add load_records, mode, and encoding arguments to read
- Documentation
  - Fix Variable attribute assignments in user guide.
  - Add more examples in docstrings.
  - Add docs for collecting the results of non-blocking NEOS Server solves.
- Testing
  - Test the special value usage in assignments for Parameter, ImplicitParameter and Operation (Sum, Smax, Smin, Product).
  - Add hansmpsge model to the model library.
  - Add tests for the new arguments of write
  - Add tests for the new arguments of read


GAMSPy 0.11.3
=============
- General
  - Fix setRecords bug
  - Run after an equation is defined
- Testing
  - Fix incorrect order of setRecords in gapmin model
  - Fix domain violation in the unit tests revealed by the execution of 
  equation definitions in immediate mode.
  - Use gams_math.sqr instead of custom sqr function in tests.


GAMSPy 0.11.2
=============
- General
  - Fix the bug in writing only modified symbols.
  - Return summary dataframe for all synchronous backends.
  - Fix the bug in using set, alias attributes in conditions.
- Documentation
  - Re-run notebooks to reflect the changes in solve summary.
- Testing
  - Add tests for the returned summary dataframe from solve.
  - Add tests for solve with trace options.


GAMSPy 0.11.1
=============
- General
  - Fix missing atttributes of Alias such as .first, .last etc.
  - Fix global option bug
  - Display summary on Jupyter Notebook.
- Testing
  - Add tests for Alias attributes.

GAMSPy 0.11.0
=============
- General
  - Generate expression representation as soon as it is created to avoid tall recursions.
  - Find variables in equations by iteratively traversing instead of doing recursion.
  - Add NEOS Server as a backend to solve models.
  - Fix domain for the equations that were specified in the constructor of the equation.
  - Check if the container of domain symbols of a symbol match with the symbol's container.
  - Check if the container is valid before running the model.
- Documentation
  - Add documentation for NEOS backend.
- Testing
  - Add NEOS Server as a backend to solve models.
  - Add tests for NEOS backend.
  - Add tests for equations that were defined in the constructor.
  - Add tests for checking the containers of domain symbols.

GAMSPy 0.10.5
=============
- General
  - Fix the issue of not setting options that are set to 0 (bug fix)
- Testing
  - Remove duplicated equations in models for MCP models.

GAMSPy 0.10.4
=============
- General
  - Fix not equals overload of Ord and Card operations (bug fix)
  - Refactor generation of GAMS string
- Documentation
  - Move doc dependencies to pyproject.toml

GAMSPy 0.10.3
=============
- General
  - Allow creating log file in working directory.
  - Forbid extra arguments for pydantic models (Options, EngineCofig)
- Documentation
  - Update model options table
  - Update jupyter notebook examples
- Testing
  - Adapt tests to new Options class instead of using dictionary.

GAMSPy 0.10.2
=============
- General
  - Write and read only dirty symbols instead of all symbols to improve performance (~30% improvement on running all model library models).
  - Make gdx file names thread safe by using uuid.
- Documentation
  - Fix api reference for inherited members.
  - Make execution modes and debugging section of container documentation a separate page.
- Testing
  - Add a new test for sending extra files to GAMS Engine.
  - Add scripts/atomic_conda_env.py to avoid race condition for parallel builds in the pipeline.

GAMSPy 0.10.1
=============
- General
  - Fix ellipsis syntax bug for variable and equation attributes
  - Introduce Pydantic as a dependency for options and engine config validation
- Documentation
  - Change reference API structure so that each class has its own page
- Testing
  - Simplify reinstall.py script
  - Add tests for options
  - Update tests for symbol creation

GAMSPy 0.10.0
=============

- Initial release.
