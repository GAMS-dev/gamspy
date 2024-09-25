--- Job _0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms Start 09/25/24 12:26:27 47.6.0 c2de9d6d LEX-LEG x86 64bit/Linux
--- Applying:
    /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/gmsprmun.txt
--- GAMS Parameters defined
    Input /tmp/tmpigwg6wc1/_0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms
    Output /tmp/tmpigwg6wc1/_0a53e5b5-083d-48f7-ae93-cefd4aa99599.lst
    ScrDir /tmp/tmpigwg6wc1/tmprswuip5s/
    SysDir /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/
    LogOption 3
    Trace /tmp/tmpigwg6wc1/_0a53e5b5-083d-48f7-ae93-cefd4aa99599.txt
    License /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/gamslice.txt
    OptDir /tmp/tmpigwg6wc1/
    LimRow 0
    LimCol 0
    TraceOpt 3
    GDX /tmp/tmpigwg6wc1/_0a53e5b5-083d-48f7-ae93-cefd4aa99599out.gdx
    ResLim 100
    SolPrint 0
    PreviousWork 1
    gdxSymbols newOrChanged
Licensee: GAMS Demo, for EULA and demo limitations see   G240530/0001CB-GEN
          https://www.gams.com/latest/docs/UG%5FLicense.html         DC0000
          /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/gamslice.txt
          Demo license for demonstration and instructional purposes only
Processor information: 1 socket(s), 12 core(s), and 16 thread(s) available
--- Starting compilation
--- _0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms(67) 4 Mb
--- Starting execution: elapsed 0:00:00.001
--- Generating LP model transport
--- _0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms(111) 4 Mb
---   6 rows  7 columns  19 non-zeroes
--- Range statistics (absolute non-zero finite values)
--- RHS       [min, max] : [ 2.750E+02, 6.000E+02] - Zero values observed as well
--- Bound     [min, max] : [        NA,        NA] - Zero values observed as well
--- Matrix    [min, max] : [ 1.260E-01, 1.000E+00]
--- Executing CPLEX (Solvelink=2): elapsed 0:00:00.001

IBM ILOG CPLEX   47.6.0 c2de9d6d Sep 12, 2024          LEG x86 64bit/Linux    

*** This solver runs with a demo license. No commercial use.
--- GMO setup time: 0.00s
--- GMO memory 0.50 Mb (peak 0.50 Mb)
--- Dictionary memory 0.00 Mb
--- Cplex 22.1.1.0 link memory 0.00 Mb (peak 0.00 Mb)
--- Starting Cplex

Version identifier: 22.1.1.0 | 2022-11-28 | 9160aff4d
CPXPARAM_Advance                                 0
CPXPARAM_Simplex_Display                         2
CPXPARAM_MIP_Display                             4
CPXPARAM_MIP_Pool_Capacity                       0
CPXPARAM_TimeLimit                               100
CPXPARAM_MIP_Tolerances_AbsMIPGap                0
Tried aggregator 1 time.
LP Presolve eliminated 0 rows and 1 columns.
Reduced LP has 5 rows, 6 columns, and 12 nonzeros.
Presolve time = 0.00 sec. (0.00 ticks)

Iteration      Dual Objective            In Variable           Out Variable
     1              73.125000    x(seattle,new-york) demand(new-york) slack
     2             119.025000     x(seattle,chicago)  demand(chicago) slack
     3             153.675000    x(san-diego,topeka)   demand(topeka) slack
     4             153.675000  x(san-diego,new-york)  supply(seattle) slack

--- LP status (1): optimal.
--- Cplex Time: 0.00sec (det. 0.01 ticks)


Optimal solution found
Objective:          153.675000

--- Reading solution for model transport
--- Executing after solve: elapsed 0:00:00.009
--- _0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms(173) 4 Mb
--- GDX File /tmp/tmpigwg6wc1/_0a53e5b5-083d-48f7-ae93-cefd4aa99599out.gdx
*** Status: Normal completion
--- Job _0a53e5b5-083d-48f7-ae93-cefd4aa99599.gms Stop 09/25/24 12:26:27 elapsed 0:00:00.009
