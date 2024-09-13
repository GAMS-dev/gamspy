--- Job _4026f03e-2913-4c5f-b311-ad396d75a6de.gms Start 09/13/24 12:03:51 47.6.0 c2de9d6d LEX-LEG x86 64bit/Linux
--- Applying:
    /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/gmsprmun.txt
--- GAMS Parameters defined
    Input /tmp/tmpfxuujkno/_4026f03e-2913-4c5f-b311-ad396d75a6de.gms
    Output /tmp/tmpfxuujkno/_4026f03e-2913-4c5f-b311-ad396d75a6de.lst
    ScrDir /tmp/tmpfxuujkno/tmpiha7id9o/
    SysDir /home/muhammet/anaconda3/envs/py38/lib/python3.8/site-packages/gamspy_base/
    LogOption 3
    Trace /tmp/tmpfxuujkno/_4026f03e-2913-4c5f-b311-ad396d75a6de.txt
    License /home/muhammet/.local/share/GAMSPy/gamspy_license.txt
    OptDir /tmp/tmpfxuujkno/
    LimRow 0
    LimCol 0
    TraceOpt 3
    GDX /tmp/tmpfxuujkno/_4026f03e-2913-4c5f-b311-ad396d75a6deout.gdx
    ResLim 100
    SolPrint 0
    PreviousWork 1
    gdxSymbols newOrChanged
Licensee: GAMSPy Incremental Professional                G240510+0003Cc-GEN
          GAMS                                                       DC0000
          /home/muhammet/.local/share/GAMSPy/gamspy_license.txt
          node:88412201                                                    
          Evaluation license: Not for commercial or production use
          The evaluation period of the license will expire on May 14, 2029
Processor information: 1 socket(s), 12 core(s), and 16 thread(s) available
--- Starting compilation
--- _4026f03e-2913-4c5f-b311-ad396d75a6de.gms(67) 4 Mb
--- Starting execution: elapsed 0:00:00.000
--- Generating LP model transport
--- _4026f03e-2913-4c5f-b311-ad396d75a6de.gms(111) 4 Mb
---   6 rows  7 columns  19 non-zeroes
--- Range statistics (absolute non-zero finite values)
--- RHS       [min, max] : [ 2.750E+02, 6.000E+02] - Zero values observed as well
--- Bound     [min, max] : [        NA,        NA] - Zero values observed as well
--- Matrix    [min, max] : [ 1.260E-01, 1.000E+00]
--- Executing CPLEX (Solvelink=2): elapsed 0:00:00.001

IBM ILOG CPLEX   47.6.0 c2de9d6d Sep 12, 2024          LEG x86 64bit/Linux    

--- GAMS/CPLEX licensed for continuous and discrete problems.
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
--- Executing after solve: elapsed 0:00:00.011
--- _4026f03e-2913-4c5f-b311-ad396d75a6de.gms(173) 4 Mb
--- GDX File /tmp/tmpfxuujkno/_4026f03e-2913-4c5f-b311-ad396d75a6deout.gdx
*** Status: Normal completion
--- Job _4026f03e-2913-4c5f-b311-ad396d75a6de.gms Stop 09/13/24 12:03:51 elapsed 0:00:00.011
