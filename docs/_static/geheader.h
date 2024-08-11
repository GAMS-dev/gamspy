/*
  Include file for GAMS external equation interface

  Utility routines for writing to the status and log files.

  The strings that are passed back to GAMS are stored in part of the
  control vector, Icntr. The packing convention is as follows:

  Icntr[I_BufStrt]:
     points to the first position in Icntr available as a
     a buffer. It is defined by the Solver/GAMS and it should
     not be changed.  It uses Fortran (i.e. 1-based) indexing.
  Icntr[I_Length]:
     Holds the length of Icntr. All space from Icntr[I_BufStrt]
     to Icntr[I_Length] is available for the buffer.
  Icntr[I_BufLen]:
     holds the number of Integer positions in Icntr already in use.
     It is always zero when GEFUNC is called and is accumulated
     in these routines. The first free position in the buffer is
     therefore always accessed as
     Icntr[Icntr[I_BufStrt]+Icntr[I_BufLen]-1].

  Records are stored in the buffer with a two-integer header record
  followed by the characters packed into integers. The first integer
  in the header holds the number of characters in the following
  string. The second integer is used to tell where the string should
  go: 1 means the Status file and 2 means the Log file. Other value
  can later be used to indicate other distinations.
  The strings themselves are converted to chars packed sizeof(int) to
  one int position.

  To avoid problems with mixing Fortran and C, strings should not
  contain any null characters or new-line characters. And because
  the string may be shifted a few characters right it should also
  not contain tab characters. Each string is interpreted as one
  output line.

  If there is insufficient space in the buffer then the last part
  is silently ignored. The header records reflect what is actually
  stored and not what we would have like to store.

  Icntr[I_Debug] is used to turn debugging on. If Icntr[I_Debug] is nonzero
             then a file with the name debugext.txt is opened and
             all communication  via GEstat and GElog is written
             to this file. The file is flushed continuously to avoid
             loosing the last interesting part.

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#  if defined(GE_EXPORTS)
#    define GE_API __declspec(dllexport)
#  else
#    define GE_API __declspec(dllimport)
#  endif
#  define GE_CALLCONV __stdcall
#elif defined(__GNUC__)
#  if defined(GE_EXPORTS)
  #  define GE_API __attribute__((__visibility__("default")))
#  else
#    define GE_API
#  endif
#  define GE_CALLCONV
#else
#  define GE_API
#  define GE_CALLCONV
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void (GE_CALLCONV * msgcb_t)
     (const int *mode, const int *nchars, const char *buf, int len);

typedef void (GE_CALLCONV * msgcb2_t)
     (void *usrmem, const int *mode, const int *nchars, const char *buf, int len);

GE_API int GE_CALLCONV
gefunc (int *icntr, double *x, double *f, double *d, msgcb_t msgcb);

GE_API int GE_CALLCONV
gefunc2 (int *icntr, double *x, double *f, double *d, msgcb2_t msgcb, void *usrmem);

#define TOSTAT 1
#define TOLOG 2

#define DOINIT       1
#define DOTERM       2
#define DOEVAL       3
#define DOCONSTDERIV 4
#define DOHVPROD     5

/* these next are offsets in icntr for various values */
#define I_Length   0
#define I_Neq      1
#define I_Nvar     2
#define I_Nz       3
#define I_Mode     4
#define I_Eqno     5
#define I_Dofunc   6
#define I_Dodrv    7
#define I_Newpt    8
#define I_Getfil   9                    /* set non-zero by external module to
                                         * request a string */
#define I_Smode   12                    /* set non-zero by solver to indicate
                                         * a string is loaded in the buffer */
#define I_ConstDeriv 13                 /* 1=true indicates user can return
                                         * constant derivatives */
#define I_HVProd  14                    /* 1=true indicates user can return
                                         * Hessian-vector products */
#define I_BufStrt 25                    /* Fortran-based!! */
#define I_BufLen  26
#define I_Debug   27

/* values used in the string communication  between solver and external module */
#define I_Scr     11                    /* scratch directory */
#define I_Wrk     12                    /* working directory */
#define I_Sys     13                    /* GAMS system directory */
#define I_Cntr    14                    /* control file */

#define LOGFILE   1
#define STAFILE   2
/* #define LSTFILE   4 */

#define INTS(x) ((((x)-1) / (int)sizeof(int)) + 1)

static FILE *fpDebug = NULL;

static void
GEwrite (int *icntr, char *line, int mode)
{
  int ints2write;
  int nBytes;
  int startMsg;
  /*
    If in debug mode, write the string to debugext.txt
  */

  if ( icntr[I_Debug] ) {
    if (NULL == fpDebug) {
      fpDebug = fopen ("debugext.txt", "w");
    }
    if (NULL == fpDebug) {
      fprintf (stdout, "Can't create debugext.txt\n");
    }
    else {
      if (mode == TOSTAT) fprintf (fpDebug,"Stat: %s\n",line);
      if (mode == TOLOG)  fprintf (fpDebug," Log: %s\n",line);
      fflush (fpDebug);
    }
  }

  /* leaves space for two control ints before this */
  startMsg   = icntr[I_BufStrt] + icntr[I_BufLen] + 1;
  /* Length in sizeof(int)-byte words after fixed part and control ints */
  ints2write = icntr[I_Length] - startMsg;

  if (ints2write <= 0)
    return;                             /* Not enough space for anything */
  nBytes = strlen(line);
  if (nBytes > 256)
    nBytes = 256;                       /* Max line length = 256 */
  if ( INTS(nBytes) > ints2write)
    nBytes = ints2write*sizeof(int);    /* line doesn't fit */

  memcpy((char *)(icntr+startMsg),line,nBytes);
  icntr[startMsg-2] = nBytes;           /* Length of string */
  icntr[startMsg-1] = mode;             /* Goes to status or log file (depending on mode) */
  icntr[I_BufLen] += 2 + INTS(nBytes);        /* Number of ints consumed */

  return;
}

/* provide prototypes to shut up compiler warnings */
void
GEstat(int *icntr, char *line);
void
GElog(int *icntr, char *line);
int
GEname (const int *icntr, char *buf, int bufLen);

void
GEstat(int *icntr, char *line)
{
  GEwrite(icntr, line, TOSTAT);
  return;
}

void
GElog(int *icntr, char *line)
{
  GEwrite(icntr, line, TOLOG);
  return;
}

/* routine to get a filename from the control vector
 * it works similar to the WINAPI function GetModuleName, etc:
 *
 *  buf [out]    : pointer to buffer for name.  If the buffer is too small,
 *                 the string copied to it is truncated,
 *                 and no null byte is written
 *  bufLen [in]  : the length, in chars, of the name buffer
 *
 *  return:
 *     success   : return the number of chars copied to the buffer,
 *                 exclusive of the terminating null.
 *     failure   : -1
 */
int
GEname (const int *icntr, char *buf, int bufLen)
{
  int nChars;                           /* number of chars to copy */
  char *from;

  nChars = icntr[11];
  if (nChars < 0) {
    return -1;
  }
  if (nChars > bufLen) {
    nChars = bufLen;
  }
  from = (char *) (icntr + icntr[10] - 1);
  strncpy (buf, from, nChars);
  if (nChars < bufLen) {
    buf[nChars] = '\0';
  }

  return nChars;
} /* GEname */


#ifdef __cplusplus
}
#endif
