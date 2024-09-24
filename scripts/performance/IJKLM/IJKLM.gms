$eval start jnow
option limrow=0, limcol=0, solprint=off, reslim=0;

set i,j,k,l,m, IJK(i,j,k), JKL(j,k,l), KLM(k,l,m);
Variable z, x(i,j,k,l,m);
Equations obj, ei;

obj.. z =e= 1;
ei(i).. sum((IJK(i,j,k),JKL(j,k,l),KLM(k,l,m)), x(i,j,k,l,m)) =g= 0;

model mi /obj, ei/;

$if not set GDXdata $set GDXdata 'scripts/performance/IJKLM/data/data.gdx'
$gdxLoad '%GDXdata%' i,j,k,l,m

$if not set R $set R 1
$if not set N $set N 1

Set r /1*%R%/, n /1*%N%/; Parameter t(r); Scalar fix, startn;

fix = jnow - %start%;
loop (r, 
    startn = jnow;
    loop (n, 
        execute_load '%GDXdata%', IJK,JKL,KLM;
        $$if not set solve mi.JustScrDir = 1
        solve mi minimizing z using lp;
    );
    t(r) = ((fix + jnow - startn) * 24 * 3600) / card(n);
);

execute_unload 'scripts/performance/IJKLM/results/result.gdx', t;
