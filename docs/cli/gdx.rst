gamspy gdx
==========

Allows dumping and comparing GDX files.

Usage
-----

::

  gamspy gdx dump <file> [OPTIONS]

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - --V / --Version
     - -v
     - False
     - Write version info of input file only
   * - --output
     - -o
     - None
     - Write output to file
   * - --symb
     - -s
     - None
     - Select a single identifier
   * - --ueltable
     - -u
     - None
     - Include all unique elements
   * - --delim
     - -d
     - None
     - Specify a dimension delimiter (period, comma, tab, blank, semicolon)
   * - --decimalsep
     - -p
     - None
     - Specify a decimal separator (period, comma)
   * - --noheader
     - -H
     - False
     - Suppress writing of the headers
   * - --nodata
     - -D
     - False
     - Write headers only; no data
   * - --csvallfields
     - -a
     - False
     - Write all variable/equation fields in CSV
   * - --csvsettext
     - -t
     - False
     - Write set element text in CSV
   * - --symbols
     - -S
     - False
     - Get a list of all symbols
   * - --domaininfo
     - -i
     - False
     - Get list of all symbols showing domain information
   * - --symbolsasset
     - -A
     - False
     - Get all symbols as data for a set
   * - --symbolsassetdi
     - -B
     - False
     - Get symbols as set with domain info
   * - --settext
     - -T
     - False
     - Show associated set text
   * - --format
     - -f
     - None
     - Output format (normal, gamsbas, csv)
   * - --dformat
     - -F
     - None
     - Data format (normal, hexponential, hexBytes)
   * - --cdim
     - -c
     - None
     - Use last dimension as CSV column headers (Y/N)
   * - --filterdef
     - -x
     - None
     - Filter default values (Y/N)
   * - --epsout
     - -e
     - None
     - String for EPS values
   * - --naout
     - -n
     - None
     - String for NA values
   * - --pinfout
     - 
     - None
     - String for Positive Infinity values
   * - --minfout
     - 
     - None
     - String for Negative Infinity values
   * - --undfout
     - 
     - None
     - String for Undefined values
   * - --zeroout
     - 
     - None
     - String for Zero values
   * - --header
     - 
     - None
     - New header for CSV output

::

  gamspy gdx diff <file1> <file2> [OPTIONS]

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - Option
     - Short
     - Default
     - Description
   * - --eps
     - -e
     - None
     - Epsilon value for comparison
   * - --releps
     - -r
     - None
     - Relative epsilon value for comparison
   * - --field
     - -f
     - None
     - Field to compare: L, M, Up, Lo, Prior, Scale, All
   * - --fldonly
     - -o
     - False
     - Write variable/equation as parameter for selected field
   * - --diffonly
     - -d
     - False
     - Write variable/equation as parameter with extra field dimension
   * - --cmpdefaults
     - -c
     - False
     - Compare default values
   * - --cmpdomains
     - -m
     - False
     - Compare domain information
   * - --matrixfile
     - -x
     - False
     - Compare GAMS matrix files in GDX format
   * - --ignoreorder
     - -i
     - False
     - Ignore UEL order of input files
   * - --setdesc
     - -s
     - Y
     - Compare explanatory texts for set elements (Y/N)
   * - --id
     - -I
     - None
     - One or more identifiers to compare
   * - --skipid
     - -S
     - None
     - One or more identifiers to skip

Examples
--------

Dump a GDX file::

  $ gamspy gdx dump gdxfile.gdx

Compare two GDX files::

  $ gamspy gdx diff gdxfile.gdx gdxfile2.gdx