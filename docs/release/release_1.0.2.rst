GAMSPy 1.0.2
------------

Release Date: 01.10.2024

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
  