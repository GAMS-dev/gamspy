GAMSPy 1.2.0
------------

Release Date: 19.11.2024

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
  