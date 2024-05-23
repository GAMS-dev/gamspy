$title Linear Phase Lowpass Filter Design (FDESIGN,SEQ=379)

$onText
This model finds the filter weights for a finite impulse response
(FIR) filter. We use rotated quadratic cones for the constraints.

This model is the minimax linear phase lowpass filter design from Lobo
et. al (Section 3.3) We model the nonlinear term 1/t in the model as
follows: introduce variables u,v, where v = 2 (and u = 1/t). Then 1/t
can be modeled as the quadratic cone

              ||[v, u-t]|| <= u+t,   u,t >=0

Contributed by Michael Ferris, University of Wisconsin, Madison


Lobo, M S, Vandenberghe, L, Boyd, S, and Lebret, H, Applications of
Second Order Cone Programming. Linear Algebra and its Applications,
Special Issue on Linear Algebra in Control, Signals and Image
Processing. 284 (November, 1998).

Keywords: quadratic constraint programming, second order cone programming,
          engineering, finite impulse response filter designment
$offText

* N2 is half the length of the FIR filter (i.e. number of discretization points)
$if not set N2 $set N2 10

Scalar n2 / %N2% /, beta / 0.01 /, step, n, omega_s, omega_p;
n       = 2*n2;
step    =   pi/180;
omega_s = 2*pi/3;
omega_p =   pi/2;

Set
   i             /   0*179  /
   omega_stop(i) / 120*179  /
   omega_pass(i) /   0* 90  /
   k             /   0*%N2% /;

Parameter omega(i);
omega(i) = (ord(i) - 1)*step;

Parameter cosomega(i,k);
cosomega(i,k) = cos((ord(k) - 1 - (n-1)/2)*omega(i));
* correction for when cos()=0 was missed due to rounding errors
cosomega(i,k)$(abs(cosomega(i,k))<1e-13) = 0;

Variable
   h(k)
   t
   v2 'for conic variable u - t'
   v3 'for conic variable u + t';

Positive Variable u, v, v3;

Equation
   passband_up_bnds(i)
   cone_lhs, cone_rhs
   so
   passband_lo_bnds(i)
   stopband_bnds(i)
   stopband_bnds2(i);

passband_up_bnds(i)$omega_pass(i)..
   2*sum(k$[ord(k) < card(k)], h(k)*cosomega(i,k)) =l= t;

cone_rhs.. v2 =e= u - t;

cone_lhs.. v3 =e= u + t;

* Explicit cone syntax for MOSEK
* so.. v3 =c= v + v2;

so.. sqr(v3) =g= sqr(v) + sqr(v2);

passband_lo_bnds(i)$omega_pass(i)..
   u =l= 2*sum(k$[ord(k) < card(k)], h(k)*cosomega(i,k));

stopband_bnds(i)$omega_stop(i)..
  -beta =l= 2*sum(k$[ord(k) < card(k)], h(k)*cosomega(i,k));

stopband_bnds2(i)$omega_stop(i)..
   2*sum(k$[ord(k) < card(k)], h(k)*cosomega(i,k)) =l= beta;

t.lo = 1;
v.fx = 2;

Model fir_socp / all /;

solve fir_socp using qcp minimizing t;

Scalar minimax;
if(t.l > 0, minimax = 20*log10(t.l));

display minimax, h.l, t.l, u.l;
