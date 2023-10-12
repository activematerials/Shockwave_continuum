#ifndef CRshockwaveH
#define CRshockwaveH

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "fileutils.h"
#include "stringutils.h"
#include "graphicstools.h"

using namespace std;

//===============================================================================================================================================
// defining constants & definitions

#define PI      3.1415926535897932384626433832795f
#define TWOPI   6.2831853071795864769252867665590f
#define FOURPI 12.5663706143591729538505735331180f
#define sgn(a) ((a)<0.0f?(-1.0f):(1.0f))
#define k_INDEX(i,L) ((i)<=((L)>>1)?(i):((i)-(L)))
#define EPS 1e-6

//-----------------------------------------------------------------------------------------------------------------------------------------------

#define FASCII     0
#define FBINARY    1
#define FXDMF      2

#define DSTR(d,s2,s3) ((d==2)?(string(s2)):(string(s3)))

//-----------------------------------------------------------------------------------------------------------------------------------------------

#define TWO_N32 0.232830643653869628906250e-9

#ifdef DBLPREC
 #define MAXT 256
 typedef double REAL;
 #define REAL_MIN             2.2250738585072014E-308   //2^-1022
 #define REAL_MAX             1.7976931348623157081E308  //(2-2^-52)*2^1023
#else
 #define MAXT 512
 typedef float REAL;
 #define REAL_MIN             1.17549435E-38F          //2^-126
 #define REAL_MAX             3.4028234663852885981E38F  //(2-2^-23)*2^127
#endif

//===============================================================================================================================================


typedef struct {
    REAL re;
    REAL im;
} COMPLEX;

typedef struct {
    REAL x,y;
} Vector2D;

typedef struct {
    REAL x,y,z;
} Vector3D;

struct saru_state {
    unsigned int l, w; //LCG and OWS states
};

typedef struct {
    int dev;

    dim3 SGRID,SBLOCK,PGRID,PBLOCK;

    int memC,memR,memI;

    Vector2D *v,*u,*hrho,*mu;
    
    Vector2D *tp;

    Vector2D *tmp1,*tmp2,*tmp3,*tmp4;

    REAL *FFTkern_hroh,*FFTkern_u,*FFTkern_v,*FFTkern_mu;

    cufftHandle fftPlan;
    bool FFTinplace;
} GPUdata;

typedef struct {
    int format;
    int Tskip,Tint;
    bool partden,partv,partvort,fluidv,mu,h,wellCF;
    string prefix,path;
    fHandle gout,lout,tout; //global & local in time output files (for e.g. XMDF gout is the xml file containing all frame info)
} outputctrl;

typedef struct {
    int type;
    int N1,N2,N3,N4;
    REAL p1,p2,p3,p4,p5;
} conforce;


typedef struct { //simulation parameters for  solver ( diffusion constants are used mostly by Fourier part of solver)
    REAL alpha,beta,nu,xi,gamma,eta,kappa,delta,rho0,h0;
    REAL De,D,Dm,Dr; //onnly Dr is used by real space solver
    REAL p0,ap,hp; //pressure parameter: p0: repulsion for low p (includes  pressure), ap:attractive potential at intermediate densities, hp: hardcore potential
} simparams;

typedef struct {
    int RS,type,step;
    REAL np1,np2;
} noiseinfo;

class CSpinners {
 public:
    CSpinners();
    ~CSpinners();

    int initparams(int n, char* argv[]);
    int initialize();

    void startoutput(); //create global out file
    void frameoutput(int framenum); //write frame info and update global output
    void stopoutput();

    void run();

private:
    int ctype,RS;
    int Nx,Ny,N,Nw,NTr;
    int Nt,mpon,mpon2,omega2on,omega2off;
    noiseinfo noise;
    
    conforce initconf;

    REAL ht,hx,hy,Lx,Ly,dkx,dky,t;
    Vector2D *v,*u,*hrho,*mu,*tp;
    REAL *tmpr;

    simparams S;
    REAL omega,omega2,lcoeff,mp,mp2,omega_org;
    conforce conf;
    GPUdata Gdata;
    outputctrl outctrl;

    int integrate(int n);

    bool cerr(const char *s="n/a");

    // Function generates Randnom number from in [0..1]
    REAL ran0() {return (1.0*rand())/(1.0*RAND_MAX);};
};


#endif
