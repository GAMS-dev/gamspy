$title Resource-Constrained Project Scheduling Problem (RCPSP,SEQ=429)

$ontext
Resource-constrained project scheduling problem (RCPSP) in a formulation
which encodes the schedule using binary finishing time indicator variables
as first proposed by Pritsker.

Problem and model formulation based on:
Pritsker, A. Alan B., Lawrence J. Waiters, and Philip M. Wolfe. "Multiproject
scheduling with limited resources: A zero-one programming approach."
Management science 16.1 (1969): 93-108.

Contains embedded Python code for parsing instance data from the classic
problem library PSPLIB from Kolisch and Sprecher.

Instance library, generator and file format:
Kolisch, Rainer, and Arno Sprecher. "PSPLIB-a project scheduling problem
library: OR software-ORSEP operations research software exchange program."
European journal of operational research 96.1 (1997): 205-216.
http://www.om-db.wi.tum.de/psplib/main.html

As default the first instance from PSPLIBs subset with 30 projects is solved to optimality (makespan=43).
$offtext

* Comment out the following dollar set statement in order to use a tiny explicitly
* specified example project instead of an instance from the PSPLIB
$set instanceName j301_1.sm

* Base project information (primary parameters)
Sets 
   j               'jobs (must be topologically ordered, i.e. i<j implies j is not a predecessor of i)'
   t               'time periods'
   r               'renewable resources';

Alias (j,i), (t,tau);

Set pred(i,j)      'yes if and only if i is predecessor of j (order relation)';

Parameters
   capacities(r)   'Renewable resource capacities available in all periods'
   durations(j)    'Job durations (processing times)'
   demands(j,r)    'Number of resource units from r the job j requires/occupies while active';

* Derived parameters (derived by GAMS from data supplied by parameters above)
Parameters
   efts(j)         'Earliest finishing times'
   lfts(j)         'Latest finishing times';

Sets    
   tw(j, t)        'yes if, and only if, t is in finish time window of job j (efts(j)<=t<=lfts(j))'
   actual(j)       'set of actual jobs (j without dummy jobs)'
   lastJob(j)      'singleton set containing only the last dummy job'
   fw(j,t,tau)     'yes if, and only, if job j (active in period t) can be finished in period tau';

Binary variable 
   x(j,t)          '1 if and only if job j finishes in period t';
Variables       
   makespan        'total project duration';

Equations
   objective       'determine makespan through finishing time of last job'
   precedence      'enforce job precedences'
   resusage        'limit consumptions of renewable resources'
   once            'each job must be scheduled exactly once';

objective..                 makespan =e= sum(j$lastJob(j), sum(t$tw(j,t), x(j,t)*(ord(t)-1)));
precedence(pred(i,j))..     sum(tw(i,t), ord(t)*x(i,t)) =l= sum(t$tw(j,t), ord(t)*x(j,t)) - durations(j);
resusage(r,t)..             sum(actual(j), demands(j,r)*sum(fw(j,t,tau), x(j,tau))) =l= capacities(r);
once(j)..                   sum(tw(j,t), x(j,t)) =e= 1;

Model rcpsp  /all/;

$ifThenE set instanceName
$onEmbeddedCode Python: %instanceName%
# Utility functions
def ints(strs): return [ int(s) for s in strs ]

def myset(prefix, cardinality):
    return [f'{prefix}{i+1}' for i in range(cardinality)]

def index_of_line(lines, substr):
    return next(i for i,line in enumerate(lines) if substr in line)

def rhs_part(lines, prefix):
    return lines[index_of_line(lines, prefix)].split(':')[1]

def succs_from_line(line): return [ f'j{j}' for j in line.split()[3:] ]

def column(lines, col, rowStart, rowCount):
    return [int(lines[rowIx].split()[col]) for rowIx in range(rowStart, rowStart+rowCount)]

# Parse data from text file
with open(str(gams.arguments)) as fp: lines = fp.readlines()
njobs = int(rhs_part(lines, 'jobs (incl. supersource'))
nres = int(rhs_part(lines, '- renewable').split()[0])
nperiods = int(rhs_part(lines, 'horizon'))
prec_offset = index_of_line(lines, 'PRECEDENCE RELATIONS:')+2
attrs_offset = index_of_line(lines, 'REQUESTS/DURATIONS')+3
caps_offset = index_of_line(lines, 'RESOURCEAVAILABILITIES')+2

jobs, res, periods = myset('j', njobs), myset('r', nres), myset('t', nperiods)
succs = { j: succs_from_line(lines[prec_offset+ix]) for ix,j in enumerate(jobs) }
durations = column(lines, 2, attrs_offset, njobs)
demands = [ ints(lines[ix].split()[3:]) for ix in range(attrs_offset, attrs_offset+njobs) ]
capacities = ints(lines[caps_offset].split())

# Fill data into GAMS sets and params
gams.set("j", jobs)
gams.set("r", res)
gams.set("t", periods)
gams.set("pred", [(i,j) for i in jobs for j in jobs if j in succs[i] ])
gams.set("durations", [ (j, durations[ix]) for ix, j in enumerate(jobs) ])
gams.set("demands", [(j,r, demands[jix][rix]) for jix,j in enumerate(jobs) for rix,r in enumerate(res)])
gams.set("capacities", [ (r, capacities[rix]) for rix,r in enumerate(res) ])
$offEmbeddedCode j r t pred durations demands capacities
$else
sets i /i1*i3/
     r /r1/,
     t /t1*t4/;
pred('i1', 'i2') = yes;
pred('i2', 'i3') = yes;
parameters
    durations /i1 0, i2 1, i3 0/
    demands /i1.r1 0, i2.r1 1, i3.r1 0/
    capacities /r1 1/;
$endIf

* First and last job are dummy jobs without duration and resource consumption
actual(j)$(1 < ord(j) and ord(j) < card(j)) = yes;
* Assumption is topological ordering of jobs, hence the last job is assigned the highest number
lastJob(j)$(ord(j) = card(j)) = yes;

* Forward computation of earliest finishing times (using precendece and durations)
efts(j)$(ord(j)=1) = 1;
loop((j,i)$pred(i,j), efts(j)=max(efts(j), efts(i)+durations(j)));

* Backward computation of latest finishing times
lfts(j) = card(T);
Scalar it, jt;
for(it=card(i) downto 1,
    loop(i$(ord(i)=it),
        for(jt=card(j) downto 1,
            loop(j$(ord(j)=jt and pred(i,j)), lfts(i)=min(lfts(i), lfts(j)-durations(j)))
        )
    )
);

* Derive set of acceptable finishing times from e/lfts boundaries
tw(j, t)$(efts(j) <= ord(t) and ord(t) <= lfts(j)) = yes;
* If job j is active in t it can finish in period tau
fw(j, t, tau)$(ord(tau)>=ord(t) and ord(tau)<=ord(t)+durations(j)-1 and tw(j,tau)) = yes;

makespan.lo = 0;
solve rcpsp using mip minimizing makespan;

* Derive starting times from binary finishing time indicators
parameter st(j) Starting time of job j in result;
loop(j, loop(t, if(x.l(j,t) = 1, st(j)=ord(t)-durations(j))));
display makespan.l, st;
