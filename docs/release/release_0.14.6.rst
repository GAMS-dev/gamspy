GAMSPy 0.14.6
-------------
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
