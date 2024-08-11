#define GE_EXPORTS
#include "geheader.h"
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <memory>
#include <string.h>
#include <sstream>
#include <cmath>

 GE_API int GE_CALLCONV
gefunc( int *icntr, double *x, double *f, double *d, msgcb_t msgcb)
{
    int  findex, dofnc, dodrv, neq, nvar, nz, rc=0;

    if ( icntr[I_Mode] == DOINIT ) {
        neq = 2; // We expect 2 external equations
        nvar = 4; // 4 variables at total
        nz = 4; // And 4 differentibles
        if ( neq != icntr[I_Neq] ) {
            GElog ( icntr, "--- Number of equations do not match");
            rc = 2;
        }
        else if (nvar != icntr[I_Nvar]) {
            GElog ( icntr, "--- Number of variables do not match");
            rc = 2;
        }
        else if (nz != icntr[I_Nz]) {
            GElog ( icntr, "--- Number of differentibles do not match");
            rc = 2;
        }
        else {
            GElog ( icntr, "--- Model has the correct size.");
            // icntr[I_Debug] = 1;
            icntr[I_ConstDeriv] = 1; // signal that we will do constant derivatives
        }
        return rc;
    }
    else if ( icntr[I_Mode] == DOTERM ) {
        // Do your cleanup here
        GElog ( icntr, "--- Terminating");
        return rc;
    }
    else if ( icntr[I_Mode] == DOEVAL ) {
        GElog ( icntr, "--- Evaluation mode");
        findex = icntr[I_Eqno];
        dofnc = icntr[I_Dofunc];
        dodrv = icntr[I_Dodrv];

        // I_Newpt can save you computation in some circumstances
        /*
        if (icntr[I_Newpt] == 1) {
            GElog ( icntr, "--- New point provided");
        }else {
            GElog ( icntr, "--- Old point again");
        }
        */

        if ( findex >= 1 && findex <= 2 ) {
            if ( dofnc ) {
                if (findex == 1)
                    *f = std::sin(x[0]) - x[2];
                else
                    *f = std::cos(x[1]) - x[3];
            }
            /*
               The vector of derivatives is needed. The derivative with respect
               to variable x(i) must be returned in d(i). The derivatives of the
               linear terms, here -Z, must be defined each time.
               */
            if ( dodrv ) {
                if(findex == 1)
                {
                    d[0] = std::cos(x[0]);
                    d[2] = -1.0;
                }
                else {
                    d[1] = -std::sin(x[1]);
                    d[3] = -1;
                }
            }

        }
        else {
            GEstat( icntr," ** fIndex has unexpected value.");
            rc = 2;
        }
        return rc;
    }
    else if ( icntr[I_Mode] == DOCONSTDERIV ) {
        GElog ( icntr, "--- Constant derivative call");
        findex = icntr[I_Eqno];
        d[findex + 1] = -1.0;
        rc = 0;
        return rc;
    }
    else {
        GElog( icntr," ** Mode not defined.");
        GEstat( icntr," ** Mode not defined.");
        rc = 2;
        return rc;
    }
}
