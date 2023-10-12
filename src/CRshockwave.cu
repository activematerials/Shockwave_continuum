/*
=========================================================================================================================================
ConrollersGPU-SW
CUDA simulation for continuum rollers - shockwave phase
(c) Andreas Glatz, 2022-2023

based on version 2.6 of general continuum roller code history:
=========================================================================================================================================
*/
#define Sversion 0x00020605


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cuda.h>
#include <time.h>

#include "fileutils.h"
#include "stringutils.h"
#include "paramfile.h"

#include "CRshockwave.h"


//===========================================================================

//Random number generator using ad-hoc 64bit states (per thread state generation) "state-free"
//3 32-bit seeds are used to mix them down to the 64-bit state
//*based on Saru: Computer Physics Communications, Volume 184, Issue 4, April 2013, Pages 1119–1128*
/*usage:
 call inside the kernel function first: G_saru_mixer3(seed1,seed2,seed3,state)
 generate random numbers in the kernel function by (advances the state as well): x=G_saru_ran(state) [type REAL]
 */


//prepare the state from 3 32-bit ints, algorithm created using "a test harness, trying hundreds of mixing functions with millions of constants"
#ifndef __CPUCODE__
__device__
#endif
inline void G_saru_mixer3(unsigned int seed1, unsigned int seed2, unsigned int seed3,saru_state &state) {
    seed3^=(seed1<<7)^(seed2>>6);
    seed2+=(seed1>>4)^(seed3>>15);
    seed1^=(seed2<<9)+(seed3<<8);
    seed3^=0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2+=0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1^=0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2+=seed1*seed3;
    seed1+=seed3 ^ (seed2>>2);
    seed2^=((signed int)seed2)>>17;
    
    state.l  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    state.w = (state.l+seed2) ^ (((signed int)state.l)>>8);
    state.l  = state.l + (state.w*(state.w^0xdddf97f5));
    state.w = 0xABCB96F7 + (state.w>>1);
};
    
//the actual PRNG, advances state and creates a REAL in (0,1]
//advance the l and w elements of the state (called by the actual PRNG)
//"uses a linear congruential generator (LCG) and an Offset Weyl Sequence (OWS) to advance the two words"
//parameter v: "Saru combines these two components in a way to obscure any regular patterns from any application. It mixes the states together and churns the output with some simple xors and a multiply."
#ifndef __CPUCODE__
__device__
#endif
inline REAL G_saru_ran(saru_state &state) {
    unsigned int v;
    
    //advance the state (even the unused intial one)
    state.l=0x4beb5d59*state.l+0x2600e1f7; //the LCG
    state.w=state.w+0x8009d14b+((((signed int)state.w)>>31)&0xda879add); //the OWS
    
    //"This mixing (in just two lines of code) seems ad-hoc, but actually it is the end result of intensive experiments."
    v=(state.l ^ (state.l>>26))+state.w;
    v=(v^(v>>20))*0x6957f5a7;
    
#ifdef __DBLPREC__
    //for double we use v and state.l to generate the 52 bits for the mantissa
    return (((signed int)v)*TWO_N32+(0.5+0.5*TWO_N32))+((signed int)state.l)*(TWO_N32*TWO_N32);
#else
    //for float we only need v, we can drop 1 bit since floats have a 23 bit mantissa
    return ((signed int)(v>>1))*(1.0f/0x80000000);
#endif
};


//====================================================================================================

__global__ void G_rarraymul2Dvec(int N,Vector2D *v,REAL *x)
{
    int i=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    Vector2D vi;
    REAL xi;
    if(i<N)
    {
        vi=v[i];
        xi=x[i];
        vi.x=xi*vi.x;vi.y=xi*vi.y;
        v[i]=vi;
    }
};
    
//====================================================================================================
// confinement potential
    
    
#ifndef __CPUCODE__
__device__
#endif
inline void G_getCF(int i,int j,int Nx,int Ny,REAL hx,REAL hy,conforce conf,Vector2D &CF,bool &nopart) { //ui not passed, add if needed
    Vector2D r,mr,hpos,dr;
    REAL a,b,x,dis,idis;
    int n1,n2,m;
    
    CF.x=CF.y=0.0;
    nopart=false;
    if(conf.type>0) {
        if(conf.type==1) { //confinement +
            if((i>0.15*Nx && i<0.85*Nx) && (j>0.15*Ny && j<0.85*Ny)) {
                
                CF.x=conf.p1*hx*(i-Nx/2);
                CF.y=conf.p1*hy*(j-Ny/2);
            }
        }
    }
};
    

//====================================================================================================

//calculate 2D FFT kernels for given laplace coefficient dcoeff and linear term lcoeff (for u,v,hrho)
#ifndef __CPUCODE__
__global__ void G_init_FFTkern_2D(int Nx,int Ny,REAL ht,REAL dkx,REAL dky,REAL dcoeff,REAL lcoeff,REAL* kern) {
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
#else
    inline void G_init_FFTkern_2D(int idx,int Nx,int Ny,REAL ht,REAL dkx,REAL dky,REAL dcoeff,REAL lcoeff,REAL* kern) {
#endif
    int i, j;
    REAL dkxy,a;
    int N=Nx*Ny;
    if(idx<N)
    {
        i=idx%Nx;
        j=idx/Nx;
        i=k_INDEX(i,Nx);
        j=k_INDEX(j,Ny);

        a=i*dkx;dkxy=a*a;
        a=j*dky;dkxy+=a*a;

        kern[idx] = exp(-ht*(dcoeff*dkxy + lcoeff))/(1.0*N);
    }
};

//====================================================================================================

#ifndef __CPUCODE__
__global__ void G_real_step_hrho(int Nx,int Ny,REAL ht,REAL hx,REAL hy,conforce conf,Vector2D *v,Vector2D *u,Vector2D *hrho,Vector2D *out) {
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
#else
inline void G_real_step_hrho(int idx,int Nx,int Ny,REAL ht,REAL hx,REAL hy,conforce conf,Vector2D *v,Vector2D *u,Vector2D *hrho,Vector2D *out) {
#endif
    int i, j, ip, jp, im, jm, kr, kl, ku, kd;
    Vector2D hr,hrp,hrm,res,CF;
    bool nopart;
    REAL a,b;

    if(idx<Nx*Ny)
    {
        i = idx%Nx;
        ip = i + 1; if(ip==Nx) ip=0;
        im = i - 1; if(im < 0) im = Nx-1;
        j = idx/Nx;
        jp = j + 1; if(jp ==Ny) jp = 0;
        jm = j - 1; if(jm < 0) jm = Ny-1;

        kr = ip + Nx*j;   kl = im + Nx*j;
        ku = i + Nx*jp;   kd = i + Nx*jm;

        a=0.5/hx;
        b=0.5/hy;

        hr=hrho[idx]; //current values for h,rho
        hrp=hrho[kr];
        hrm=hrho[kl];

        //x derivatives
        res.x=a*(hrp.x*v[kr].x-hrm.x*v[kl].x);
        res.y=a*(hrp.y*u[kr].x-hrm.y*u[kl].x);

        hrp=hrho[ku];
        hrm=hrho[kd];

        //y derivaties
        res.x+=b*(hrp.x*v[ku].y-hrm.x*v[kd].y);
        res.y+=b*(hrp.y*u[ku].y-hrm.y*u[kd].y);

        
        res.x=hr.x-ht*res.x;
        res.y=hr.y-ht*res.y;
       
        out[idx]=res;
    }
};

//====================================================================================================
//update u & v in place

#ifndef __CPUCODE__
    //REAL alpha,REAL beta,REAL gamma,REAL delta,REAL eta,REAL kappa,REAL mp,REAL nu,REAL xi,REAL Dr,REAL rho0
__global__ void G_real_step_uv_mu(int Nx,int Ny,REAL hx,REAL hy,REAL ht,int tstep,simparams S,conforce conf,noiseinfo noise,Vector2D *v,Vector2D *u,Vector2D *hrho,Vector2D *mu,Vector2D *vnew,Vector2D *unew,Vector2D *munew) {
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
#else
inline void G_real_step_uv_mu(int idx,int Nx,int Ny,REAL hx,REAL hy,REAL ht,int tstep,simparams S,conforce conf,noiseinfo noise,Vector2D *v,Vector2D *u,Vector2D *hrho,Vector2D *mu,Vector2D *vnew,Vector2D *unew,Vector2D *munew) {
#endif
    int i, j, ip, jp, im, jm, kr, kl, ku, kd;
    REAL a,b,u2,lu,c,d,mu2,rc,omega; //lp,lm
    Vector2D hr,vi,ui,vn,un,gh,gp,vgv,vp,vm,hrr,hrl,hru,hrd,divM,mui,mun,mup,mum,grho,CF; //outflow
    saru_state sast;
    bool nopart;
    
    if(idx<Nx*Ny)
    {
        i = idx%Nx;  ip = i + 1; if(ip==Nx) ip=0;
        im = i - 1; if(im < 0) im = Nx-1;
        j = idx/Nx;  jp = j + 1; if(jp ==Ny) jp = 0;
        jm = j - 1; if(jm < 0) jm = Ny-1;

        kr = ip + Nx*j;   kl = im + Nx*j;
        ku = i + Nx*jp;   kd = i + Nx*jm;

        hr=hrho[idx]; //local h,rho
        mui=mu[idx];  //local mu
        vi=v[idx];
        ui=u[idx];

        hrr=hrho[kr];
        hrl=hrho[kl];
        hru=hrho[ku];
        hrd=hrho[kd];

        a=0.5/hx;
        b=0.5/hy;
        //grad(h)
        gh.x=a*(hrr.x-hrl.x);
        gh.y=b*(hru.x-hrd.x);
        //grad(p) -- steric repulsion, pressure = 1/(rho_c-rho)
        
        //grad(rho)
        grho.x=a*(hrr.y-hrl.y);
        grho.y=b*(hru.y-hrd.y);
        
        
        //(v*grad)v
        vp=v[kr];vm=v[kl];
        vgv.x=vi.x*a*(vp.x-vm.x);
        vgv.y=vi.x*a*(vp.y-vm.y);
        omega=-a*(vp.y-vm.y); //vorticity
        vp=v[ku];vm=v[kd];
        vgv.x=vgv.x+vi.y*b*(vp.x-vm.x);
        vgv.y=vgv.y+vi.y*b*(vp.y-vm.y);
        omega=omega+b*(vp.x-vm.x);
        c=S.beta*hr.y;
        vn.x=vi.x+ht*(c*ui.x-S.alpha*gh.x-vgv.x); //-d*vi.x
        vn.y=vi.y+ht*(c*ui.y-S.alpha*gh.y-vgv.y); //-d*vi.y

        mup=mu[kr];mum=mu[kl];
    
        divM.x=a*0.5*(3*mup.x*mup.x+mup.y*mup.y-4*hrr.y*hrr.y-3*mum.x*mum.x-mum.y*mum.y+4*hrl.y*hrl.y);
        divM.y=a*(mup.x*mup.y-mum.x*mum.y);


        mup=mu[ku];mum=mu[kd];
    
        divM.x=divM.x+b*(mup.x*mup.y-mum.x*mum.y);
        divM.y=divM.y+b*0.5*(3*mup.y*mup.y+mup.x*mup.x-4*hru.y*hru.y-3*mum.y*mum.y-mum.x*mum.x+4*hrd.y*hrd.y);

    


        c=S.gamma*hr.y-S.Dr;
        u2=ui.x*ui.x+ui.y*ui.y;
        mu2=mui.x*mui.x+mui.y*mui.y;
        lu=1.0/(sqrt(u2)+0.01);
        d=0.33333333333333*lu; // /3.0

        
        
        //confinement forces
        G_getCF(i,j,Nx,Ny,hx,hy,conf,CF,nopart);
        
        if(S.delta>1e-4) {c=-c;S.xi=-S.xi;} //make equation subcritical
        
        //pressure terms propto grad(rho)
        gp.x=S.p0; //-0.1-mp; long-range repulsion+magnetic pressure
        gp.y=S.p0;
        
        gp.x+=S.ap*hr.y; //attraction; def: S.ap=0.04
        gp.y+=S.ap*hr.y;
        
        rc=2.5/S.rho0; // rho0=1 means the particles occupy 1/10 of the container

        
        if(hr.y>rc) {
            gp.x+=S.hp; //hard-core repulsion; def. S.hp=-50.0
            gp.y+=S.hp;
        }

        
        
        //check u2/mu2 scaling with hr.y
        
        un.x=ui.x+ht*(c*ui.x-(S.xi+S.delta*u2)*hr.y*u2*ui.x+S.eta*hr.y*vi.x+S.kappa*divM.x/(hr.y+1e-5)+gp.x*grho.x+5*1.0/2.0*vi.y*omega+CF.x); //-vgv.x : is 0
        un.y=ui.y+ht*(c*ui.y-(S.xi+S.delta*u2)*hr.y*u2*ui.y+S.eta*hr.y*vi.y+S.kappa*divM.y/(hr.y+1e-5)+gp.y*grho.y-5*1.0/2.0*vi.x*omega+CF.y); //-vgv.y
        
        mun.x=mui.x+ht*(c*mui.x-(S.xi+S.delta*mu2)/hr.y*mu2*mui.x+10.0*(mui.y*mui.y*ui.x-mui.x*mui.y*ui.y));
        mun.y=mui.y+ht*(c*mui.y-(S.xi+S.delta*mu2)/hr.y*mu2*mui.y+10.0*(mui.x*mui.x*ui.y-mui.x*mui.y*ui.x));
        
        
        if(noise.type>0) { //u noise
            G_saru_mixer3(idx,tstep,noise.RS,sast);
            if(noise.type==1) { //additive noise}
            un.x+=noise.np1*(2.0*G_saru_ran(sast)-1.0);
            un.y+=noise.np1*(2.0*G_saru_ran(sast)-1.0);
            } else if(noise.type==2) {//uniform rotation between -np1 & np1
                a=noise.np1*(2.0*G_saru_ran(sast)-1.0);
                b=cos(a);
                a=sin(a);
                ui=un;
                un.x=b*ui.x-a*ui.y;
                un.y=a*ui.x+b*ui.y;
            } else if(noise.type==3) {//rotation between -phi & phi, phi depends on speed^2 and np2 , limited by np1
                u2=un.x*un.x+un.y*un.y;
                if(noise.np2>noise.np1*u2) a=noise.np1;
                else a=noise.np2/u2; //the faster the particles, the less rotation
                a=a*(2.0*G_saru_ran(sast)-1.0);
                b=cos(a);
                a=sin(a);
                ui=un;
                un.x=b*ui.x-a*ui.y;
                un.y=a*ui.x+b*ui.y;
            }
        }
        
        //write updated u,v,mu to array
        vnew[idx]=vn;
        unew[idx]=un;
        munew[idx]=mun;
    }
};

inline void G_FFT(COMPLEX *zin,COMPLEX *zout,cufftHandle &fftPlan,bool fwd=true)
{
#ifdef __DBLPREC__
        if(fwd) cufftExecZ2Z(fftPlan,(cufftDoubleComplex*) zin,(cufftDoubleComplex*) zout,CUFFT_FORWARD);
        else    cufftExecZ2Z(fftPlan,(cufftDoubleComplex*) zin,(cufftDoubleComplex*) zout,CUFFT_INVERSE);
#else
        if(fwd) cufftExecC2C(fftPlan,(cufftComplex*) zin,(cufftComplex*) zout,CUFFT_FORWARD);
        else    cufftExecC2C(fftPlan,(cufftComplex*) zin,(cufftComplex*) zout,CUFFT_INVERSE);
#endif
};

//====================================================================================================
//====================================================================================================

// Evolve equations one time step on GPU
int CSpinners::integrate(int n)
{
    Vector2D *tmpV2;

    //h & rho update
    G_real_step_hrho <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,ht,hx,hy,conf,Gdata.v,Gdata.u,Gdata.hrho,Gdata.tmp1);
    if(cerr("h,rho real")) return -10;

    G_FFT((COMPLEX*)Gdata.tmp1,(COMPLEX*)Gdata.tmp2,Gdata.fftPlan);
    if(cerr("fft hr")) return -11;

    //maybe it is faster to calculate the kernel on the fly than to use globale arrays
    G_rarraymul2Dvec  <<< Gdata.SGRID, Gdata.SBLOCK >>> (N,Gdata.tmp2,Gdata.FFTkern_hroh);
    if(cerr("fftkern hr")) return -12;

    G_FFT((COMPLEX*)Gdata.tmp2,(COMPLEX*)Gdata.tmp1,Gdata.fftPlan, false);
    if(cerr("ifft hr")) return -13;
    //flip hrho and tmp1,
    tmpV2 = Gdata.tmp1; Gdata.tmp1 = Gdata.hrho; Gdata.hrho=tmpV2;

    //u & v update, using old h&rho (tmp1)
    //only noise.type>0 does produce noise, using saru PRNG; if noise is only applied every noise step, we make it negative
    if(noise.type<0) noise.type=-noise.type;
    if(noise.step>1) {if((n%noise.step)!=0) noise.type=-noise.type;}
    G_real_step_uv_mu <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,hx,hy,ht,n,S,conf,noise,Gdata.v,Gdata.u,Gdata.tmp1,Gdata.mu,Gdata.tmp2,Gdata.tmp3,Gdata.tmp4);
    if(cerr("u,v real")) return -20;

    //we do not need tmp1 (old h,rho) anymore, use it for FFT

    G_FFT((COMPLEX*)Gdata.tmp3,(COMPLEX*)Gdata.tmp1,Gdata.fftPlan);
    if(cerr("fft u")) return -21;

    G_rarraymul2Dvec  <<< Gdata.SGRID, Gdata.SBLOCK >>> (N,Gdata.tmp1,Gdata.FFTkern_u);
    if(cerr("fftkern u")) return -22;

    G_FFT((COMPLEX*)Gdata.tmp1,(COMPLEX*)Gdata.u,Gdata.fftPlan, false);
    if(cerr("ifft u")) return -23;

    G_FFT((COMPLEX*)Gdata.tmp2,(COMPLEX*)Gdata.tmp1,Gdata.fftPlan);
    if(cerr("fft v")) return -24;

    G_rarraymul2Dvec  <<< Gdata.SGRID, Gdata.SBLOCK >>> (N,Gdata.tmp1,Gdata.FFTkern_v);
    if(cerr("fftkern v")) return -25;

    G_FFT((COMPLEX*)Gdata.tmp1,(COMPLEX*)Gdata.v,Gdata.fftPlan, false);
    if(cerr("ifft v")) return -26;

    G_FFT((COMPLEX*)Gdata.tmp4,(COMPLEX*)Gdata.tmp1,Gdata.fftPlan);
    if(cerr("fft mu")) return -27;

    G_rarraymul2Dvec  <<< Gdata.SGRID, Gdata.SBLOCK >>> (N,Gdata.tmp1,Gdata.FFTkern_mu);
    if(cerr("fftkern mu")) return -28;

    G_FFT((COMPLEX*)Gdata.tmp1,(COMPLEX*)Gdata.mu,Gdata.fftPlan, false);
    if(cerr("ifft mu")) return -29;

    
    return 0;
} // end function update

//---------------------------------------------------------------------------

void CSpinners::startoutput() {
    string s,topo,geo,origin;
    int dim,i;

    if(outctrl.format==FXDMF) {
        //2D
        outctrl.gout=FileCreate(outctrl.path+outctrl.prefix+".xmf");
        topo="1 "+IntToStr(Ny)+" "+IntToStr(Nx);
        geo="0.0 "+FloatToStr(hy)+" "+FloatToStr(hx);
        origin="0.0 "+FloatToStr(-0.5*Ly)+" "+FloatToStr(-0.5*Lx);
        dim=3;

        s="<?xml version=\"1.0\" ?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n";
        s=s+"<Domain>\n<Topology name=\"system\" TopologyType=\""+DSTR(dim,"2DCoRectMesh","3DCoRectMesh")+"\" Dimensions=\""+topo+ "\">\n</Topology>\n<Geometry name=\"geo\" Type=\""+DSTR(dim,"ORIGIN_DXDY","ORIGIN_DXDYDZ")+"\">\n<!-- Origin -->\n<DataItem Format=\"XML\" Dimensions=\""+IntToStr(dim)+"\">\n"+origin+"\n</DataItem>\n<!-- DzDyDx -->\n<DataItem Format=\"XML\" Dimensions=\""+IntToStr(dim)+"\">\n"+geo+"\n</DataItem>\n</Geometry>\n";
        //Grid information, with possibility to add temporal data
        s=s+"<Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n\n";
        FileWrite(outctrl.gout,s.c_str(),s.length());
        fflush((FILE*)outctrl.gout); //force write

        //create a binary array of zeros (z-component of u,v), since the 2D XDMF does not work correctly...
        for(i=0;i<N;i++) tmpr[i]=0.0;
        outctrl.lout=FileCreate(outctrl.path+outctrl.prefix+"_zero.bin");
        FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
    }
    if(NTr>0) {
        outctrl.tout=FileCreate(outctrl.path+outctrl.prefix+"_tracer.xmf");
        s="<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"3.0\">\n";
        s=s+"<Domain Name=\"conroll\">\n<Grid Name=\"TracerParticles\" GridType=\"Collection\" CollectionType=\"Temporal\">\n\n";
        FileWrite(outctrl.tout,s.c_str(),s.length());
        
    }
};

void CSpinners::frameoutput(int framenum) {
    string s,fn,topo,ts;
    bool wfloat;
    int dim,i,j,ip,im,jp,jm,m,k,wCFt,idx,nn,nn2;
    REAL x,min,max,a,b,dx,dy,xav,yav;
    Vector2D p;
    char *spin,*spinav;
    REAL *vortw;

    wfloat=true;
#ifdef __DBLPREC__
    wfloat=false;
#endif

    fn=outctrl.prefix+"_"+IntToStrF(framenum,6); //output file name (w/o extension), NO PATH!



    if((outctrl.format==FXDMF) && (outctrl.gout!=-1)) {
        dim=3; //create artificial 3D array
        topo="1 "+IntToStr(Ny)+" "+IntToStr(Nx);
        
        s=s+"<Grid Name=\"T"+IntToStrF(framenum,6)+"\" GridType=\"Uniform\">\n<Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n<Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n";
        s=s+"<Time Value=\""+FloatToStr(t)+"\"/>\n\n";
        
        /* we could write parameter information
         s=s+"<Attribute Center=\"Grid\" Name=\"B\" Type=\"Vector\">\n<DataItem DataType=\"Float\" Dimensions=\"1 3\" Format=\"XML\" Precision=\"8\">\n";
         s=s+FloatToStr(Bx)+" "+FloatToStr(By)+" "+FloatToStr(Bz)+"\n"; //field is always 3D, independent on the system dim
         s=s+"</DataItem>\n</Attribute>\n";
         
         s=s+"<Attribute Center=\"Grid\" Name=\"pbc\" Type=\"Vector\">\n<DataItem DataType=\"UChar\" Dimensions=\"1 "+IntToStr(dim)+"\" Format=\"XML\" Precision=\"1\">\n";
         if(dim==2) s=s+IntToStr(btype&0xFF)+" "+IntToStr((btype&0xFF00)>>8)+"\n";
         else s=s+IntToStr(btype&0xFF)+" "+IntToStr((btype&0xFF00)>>8)+" "+IntToStr((btype&0xFF0000)>>16)+"\n";
         s=s+"</DataItem>\n</Attribute>\n";
         
         s=s+"<Attribute Center=\"Grid\" Name=\"Jxext\" Type=\"Scalar\">\n<DataItem DataType=\"Float\" Dimensions=\"1\" Format=\"XML\" Precision=\"8\">\n";
         s=s+FloatToStr(Jxext)+"\n";
         s=s+"</DataItem>\n</Attribute>\n";
         
         s=s+"<Attribute Center=\"Grid\" Name=\"Kx\" Type=\"Scalar\">\n<DataItem DataType=\"Float\" Dimensions=\"1\" Format=\"XML\" Precision=\"8\">\n";
         s=s+FloatToStr(KEx)+"\n";
         s=s+"</DataItem>\n</Attribute>\n";
         
         s=s+"<Attribute Center=\"Grid\" Name=\"V\" Type=\"Scalar\">\n<DataItem DataType=\"Float\" Dimensions=\"1\" Format=\"XML\" Precision=\"8\">\n";
         s=s+FloatToStr(KExdot)+"\n";
         s=s+"</DataItem>\n</Attribute>\n";
         */
        
        s=s+"\n";
        
        //copy data from GPU
        cudaMemcpy(hrho, Gdata.hrho, Gdata.memC, cudaMemcpyDeviceToHost);
        cudaMemcpy(u, Gdata.u, Gdata.memC, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, Gdata.v, Gdata.memC, cudaMemcpyDeviceToHost);
        cudaMemcpy(mu, Gdata.mu, Gdata.memC, cudaMemcpyDeviceToHost);
        
        //split and write h & rho to binary files and create xmf entries
        min=hrho[0].x;max=hrho[0].x;
        for(i=0;i<N;i++) {x=hrho[i].x;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (h=[%le;%le]) ",min,max);
        if(outctrl.h) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_h.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"h\" Center=\"Node\">\n<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_h.bin\n</DataItem>\n</Attribute>\n";
        }
        min=hrho[0].y;max=hrho[0].y;
        for(i=0;i<N;i++) {x=hrho[i].y;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (rho=[%le;%le]) ",min,max);
        if(outctrl.partden) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_rho.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"rho\" Center=\"Node\">\n<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_rho.bin\n</DataItem>\n</Attribute>\n";
        }
        
        //split and write u & v to binary files
        //particle velocity
        min=u[0].x;max=u[0].x;
        for(i=0;i<N;i++) {x=u[i].x;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (ux=[%le;%le]) ",min,max);
        if(outctrl.partv) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_ux.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"particle velocity\" Center=\"Node\" AttributeType=\"Vector\">\n";
            if(dim==2) s=s+"<DataItem Dimensions=\""+topo+" 2\" Function=\"JOIN($0, $1)\" ItemType=\"Function\">\n";
            else s=s+"<DataItem Dimensions=\""+topo+" 3\" Function=\"JOIN($0, $1, $2)\" ItemType=\"Function\">\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_ux.bin\n</DataItem>\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_uy.bin\n</DataItem>\n";
            if(dim==3) s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+" 1\">\n"+outctrl.prefix+"_zero.bin\n</DataItem>\n";
            s=s+"</DataItem>\n</Attribute>\n";
        }
        min=u[0].y;max=u[0].y;
        for(i=0;i<N;i++) {x=u[i].y;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (uy=[%le;%le]) ",min,max);
        if(outctrl.partv) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_uy.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
        }
        
        //particle velocity vorticity (2D -> only z-component of curl(u))
        if(outctrl.partvort) {
            a=0.5/hx;b=0.5/hy;
            for(j=0;j<Ny;j++) {
                jp=j+1;if(jp==Ny) jp=0;
                jm=j-1;if(jm<0) jm=Ny-1;
                m=j*Nx;
                for(i=0;i<Nx;i++) {
                    ip=i+1;if(ip==Nx) ip=0;
                    im=i-1;if(im<0) im=Nx-1;
                    tmpr[i+m]=a*(u[ip+m].y-u[im+m].y)-b*(u[i+Nx*jp].x-u[i+Nx*jm].x); //z-comp of curl(u)
                }
            }
            outctrl.lout=FileCreate(outctrl.path+fn+"_pv.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"vorticity\" Center=\"Node\">\n<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_pv.bin\n</DataItem>\n</Attribute>\n";
            if(outctrl.wellCF) {
                wCFt=-1;
                if(((conf.type==10) || (conf.type==15)) && (conf.N1>=1) && (conf.N2>=1)) wCFt=0; //no edge rows
                if(((conf.type==11) || (conf.type==16)) && (conf.N1>=3) && (conf.N2>=3)) wCFt=1; //1 edge row around
                if(wCFt>=0) {
                    dx=(1.0*Nx)/(1.0*conf.N1);
                    dy=(1.0*Ny)/(1.0*conf.N2);
                    m=Nx/conf.N1;
                    k=Ny/conf.N2;
                    a=1.0/(1.0*m*k);
                    
                    spin=new char[conf.N1*conf.N2];
                    spinav=new char[conf.N1*conf.N2];
                    vortw=new REAL[conf.N1*conf.N2];
                    for(i=0;i<conf.N1*conf.N2;i++) {spin[i]=0;spinav[i]=0;vortw[i]=0.0;}
                    
                    for (j=wCFt;j<conf.N2-wCFt;j++) {
                        jm=(int) (j*dy);
                        for (i=wCFt;i<conf.N1-wCFt;i++) {
                            im=(int) (i*dx);
                            min=max=xav=0.0;
                            for(jp=jm;jp<jm+k;jp++) {
                                for(ip=im;ip<im+m;ip++) {
                                    idx=ip; if(idx>=Nx) idx=Nx-1;
                                    if(jp>=Ny) idx+=Nx*(Ny-1);
                                    else idx+=Nx*jp;
                                    x=tmpr[idx];
                                    xav+=x;
                                    if(x>max) max=x;
                                    else if(x<min) min=x;
                                }
                            }
                            idx=i-wCFt+(conf.N1-wCFt-wCFt)*(j-wCFt);
                            xav*=a;
                            vortw[idx]=xav;
                            if(max>4.0) spin[idx]=1;
                            else if(min<-4.0) spin[idx]=-1;
                            if(xav>0.01) spinav[idx]=1;
                            else if(xav<-0.01) spinav[idx]=-1;
                            //if(a!=b): print(i,j,vmin,vmax,vav,a,b)
                        }
                    }
                    
                    m=(conf.N1-wCFt-wCFt);
                    k=(conf.N2-wCFt-wCFt);
                    xav=0.0;
                    yav=0.0;
                    for(j=0;j<k;j++) {
                        for(i=0;i<m;i++) {
                            idx=i+m*j;
                            a=b=0.0;
                            nn=nn2=0;
                            if(i>0) {
                                nn+=spin[idx-1];a+=1.0;
                                if(i>1) {nn2+=spin[idx-1];b+=1.0;}
                            }
                            if(i<m-1) {
                                nn+=spin[idx+1];a+=1.0;
                                if(i<m-2) {nn2+=spin[idx+1];b+=1.0;}
                            }
                            if(j>0) {
                                nn+=spin[idx-m];a+=1.0;
                                if(j>1) {nn2+=spin[idx-m];b+=1.0;}
                            }
                            if(j<k-1) {
                                nn+=spin[idx+m];a+=1.0;
                                if(j<k-2) {nn2+=spin[idx+m];b+=1.0;}
                            }
                            a=nn/a;
                            xav+=spin[idx]*a;
                            if((j>0) && (i>0) && (j<k-1) && (i<m-1) && (b>0.0)) {
                                yav+=(1.0*spin[idx])*nn2/b;
                            }
                        }
                    }
                    xav*=1.0/(1.0*m*k);
                    if((m>3) && (k>3)) yav*=1.0/(1.0*(m-2)*(k-2));
                    
                    outctrl.lout=FileCreate(outctrl.path+fn+"_wCF.tsv");
                    ts="# well spin distribution and NN correlation function at time "+FloatToStr(t)+"\n";
                    ts=ts+"# wells: "+IntToStr(m)+"x"+IntToStr(k)+", edge="+IntToStr(wCFt)+"\n";
                    ts=ts+"# spinCF(MMspin)="+FloatToStr(xav)+"\n";
                    ts=ts+"# spinCF_int(MMspin)="+FloatToStr(yav)+"\n";
                    ts=ts+"#n1\tn2\tMMspin\tAVspin\tAVvort\tdisc\n";
                    FileWrite(outctrl.lout,ts.c_str(),ts.length());
                    for(j=0;j<k;j++) {
                        for(i=0;i<m;i++) {
                            idx=i+m*j;
                            ts=IntToStr(i+1)+"\t"+IntToStr(j+1)+"\t"+IntToStr(spin[idx])+"\t"+IntToStr(spinav[idx])+"\t"+FloatToStr(vortw[idx]);
                            if(spin[idx]!=spinav[idx]) ts=ts+"\tX\n";
                            else ts=ts+"\t=\n";
                            FileWrite(outctrl.lout,ts.c_str(),ts.length());
                        }
                    }
                    FileClose(outctrl.lout);
                    
                    delete[] spin;
                    delete[] spinav;
                    delete[] vortw;
                }
            }
        }
        
        //fluid velocity
        min=v[0].x;max=v[0].x;
        for(i=0;i<N;i++) {x=v[i].x;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (vx=[%le;%le]) ",min,max);
        if(outctrl.fluidv) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_vx.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"fluid velocity\" Center=\"Node\" AttributeType=\"Vector\">\n";
            if(dim==2) s=s+"<DataItem Dimensions=\""+topo+" 2\" Function=\"JOIN($0, $1)\" ItemType=\"Function\">\n";
            else s=s+"<DataItem Dimensions=\""+topo+" 3\" Function=\"JOIN($0, $1, $2)\" ItemType=\"Function\">\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_vx.bin\n</DataItem>\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_vy.bin\n</DataItem>\n";
            if(dim==3) s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+" 1\">\n"+outctrl.prefix+"_zero.bin\n</DataItem>\n";
            s=s+"</DataItem>\n</Attribute>\n";
        }
        min=v[0].y;max=v[0].y;
        for(i=0;i<N;i++) {x=v[i].y;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (vy=[%le;%le]) ",min,max);
        if(outctrl.fluidv) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_vy.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
        }
        
        //magnetic moments
        min=mu[0].x;max=mu[0].x;
        for(i=0;i<N;i++) {x=mu[i].x;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (mux=[%le;%le]) ",min,max);
        if(outctrl.mu) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_mx.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
            s=s+"<Attribute Name=\"magnetic moment\" Center=\"Node\" AttributeType=\"Vector\">\n";
            if(dim==2) s=s+"<DataItem Dimensions=\""+topo+" 2\" Function=\"JOIN($0, $1)\" ItemType=\"Function\">\n";
            else s=s+"<DataItem Dimensions=\""+topo+" 3\" Function=\"JOIN($0, $1, $2)\" ItemType=\"Function\">\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_mx.bin\n</DataItem>\n";
            s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+"\">\n"+fn+"_my.bin\n</DataItem>\n";
            if(dim==3) s=s+"<DataItem Format=\"Binary\" DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+topo+" 1\">\n"+outctrl.prefix+"_zero.bin\n</DataItem>\n";
            s=s+"</DataItem>\n</Attribute>\n";
        }
        min=mu[0].y;max=mu[0].y;
        for(i=0;i<N;i++) {x=mu[i].y;tmpr[i]=x;if(x>max) max=x;if(x<min) min=x;} printf(" (muy=[%le;%le]) ",min,max);
        if(outctrl.mu) {
            outctrl.lout=FileCreate(outctrl.path+fn+"_my.bin");FileWrite(outctrl.lout,tmpr,sizeof(REAL)*N);FileClose(outctrl.lout);
        }
        
        
        s=s+"</Grid>\n\n";
        
        FileWrite(outctrl.gout,s.c_str(),s.length());
        fflush((FILE*)outctrl.gout); //force write
    }
    
    if((NTr>0) && (outctrl.tout!=-1)) {
        cudaMemcpy(tp, Gdata.tp, NTr*sizeof(Vector2D), cudaMemcpyDeviceToHost);
        for(i=0;i<NTr;i++) {p=tp[i];p.x=p.x-0.5*Lx;p.y=p.y-0.5*Ly;tp[i]=p;}
        printf("shifted %le, %le \n",-0.5*Lx,-0.5*Ly);
        outctrl.lout=FileCreate(outctrl.path+fn+"_tracer.bin");FileWrite(outctrl.lout,tp,NTr*sizeof(Vector2D));FileClose(outctrl.lout);
        
        s="<Grid Name=\"T"+IntToStrF(framenum,6)+"\" GridType=\"Uniform\">\n";
        s=s+"<Time Value=\""+FloatToStr(t)+"\" />\n";
        
        s=s+"<Topology TopologyType=\"Polyvertex\" NodesPerElement=\""+IntToStr(NTr)+"\">\n</Topology>\n";
        
        s=s+"<Geometry GeometryType=\"XY\">\n";
        s=s+"<DataItem DataType=\"Float\" Precision=\""+IntToStr(wfloat?4:8)+"\" Endian=\"Little\" Dimensions=\""+IntToStr(NTr)+" 2\" Format=\"Binary\" Seek=\"0\">\n";
        s=s+fn+"_tracer.bin"+"\n</DataItem>\n</Geometry>\n";
        
        s=s+"</Grid>\n\n";
        
        FileWrite(outctrl.tout,s.c_str(),s.length());
    }
};

void CSpinners::stopoutput() {
    string s;

    if(outctrl.gout!=-1) {
        if(outctrl.format==FXDMF) {
            //write footer
            s="</Grid>\n</Domain>\n</Xdmf>\n";
            FileWrite(outctrl.gout,s.c_str(),s.length());
        }
        FileClose(outctrl.gout);
    }
    if((NTr>0) && (outctrl.tout!=-1)) {
        s="</Grid>\n</Domain>\n</Xdmf>\n";
        FileWrite(outctrl.tout,s.c_str(),s.length());
        FileClose(outctrl.tout);
    }
};

//---------------------------------------------------------------------------

// MAIN SIMULATION loop
void CSpinners::run()
{
    time_t ST, ET;
    bool res,outf;
    int n,frame;

    REAL p0;
    
    string s;


	t=0.0;
    frame = 0;                // starting frame number

    startoutput();

    frameoutput(0); //initial frame

    ST = time(NULL);

    //file to store chirality
    //FILE *f = fopen("chirality.txt", "w");
    
    //time loop
    outf=false;
    p0=S.p0;
    for(n=0; n<Nt; n++)
    {
        //printf("time step %06d ... ",n+1);
        
        if(n==mpon) S.p0=p0+mp;
        if(n==mpon2) S.p0=p0+mp2;
        
        //we can do one frequency step/pulse (on & off with new frequencey omega2)
        if(n==omega2on) {
            omega_org=omega;
            omega=omega2;
            S.xi=4.0*omega*S.rho0;
            S.gamma=4.0*omega*S.rho0;
            S.eta=0.25*0.2*6.*omega*S.rho0/S.h0;
        } else if(n==omega2off) {
            if(omega2off>omega2on) {
                omega=omega_org;
                S.xi=4.0*omega*S.rho0;
                S.gamma=4.0*omega*S.rho0;
                S.eta=0.25*0.2*6.*omega*S.rho0/S.h0;
            }
        }
        if(((conf.type==15) || (conf.type==16)) && (conf.N3>1)) {if(conf.N3==n) {conf.p5=0.0;printf("seed force reset at time step n=%d\n",n);}}

        res=integrate(n);
        if(res!=0) printf("error %d ... ",res);

        t=(n+1)*ht; //update time

        outf=false;
        if((res==0) && (n>outctrl.Tskip) && (n%outctrl.Tint==0)) {
            frame++;
            frameoutput(frame);
            printf("written frame %04d ...\n ",frame);
            outf=true;

    }

    if(!outf) {
        frame++;
        frameoutput(frame);
        printf("final frame %04d written.\n",frame);
    }
    stopoutput();

    ET = time(NULL);
    printf("simulation completed in %ld seconds\n", ET-ST);


};

//---------------------------------------------------------------------------

//constructor and built-in default values
CSpinners::CSpinners()
{
    ctype=1;  //computation type
    
    initconf.type=0; //initial configuration
    initconf.p1=initconf.p2=initconf.p3=initconf.p4=initconf.p5=0.0; //parameters for initial configuration
    
    conf.type=0; //confinement force (potential) type
    conf.p1=1.0;conf.p2=1.0; //parameters for potential
    conf.p3=conf.p4=conf.p5=0.0;
    conf.N1=conf.N2=conf.N3=conf.N4=0;
    
    
    Nx=1024;
    Ny=1024;
    N=Nx*Ny;
    
    NTr=0; //number of tracer particles

    Lx=128.0;
    Ly=128.0;

    Nt=500000/4*4;
    ht=0.004;

    RS = 99999;                // random seed
    
    noise.type=0;
    noise.RS=RS;
    noise.step=1;
    noise.np1=0.0;
    noise.np2=1.0;
    
//physical parameters
    conf.p1=0.0; //confinement ||  0.0375 for reversal (type=1)
    omega=0.75; //0.425; //field frequency (1=60Hz)  ||  0.75 in the figure
    S.h0=1.0; //1.0 //fluid depth (1=10 x particle radii)
    S.rho0=0.625; //.6; //particle number density  // 0.425 in the figure
    S.De=0.2; //diffusion (h,rho)
    S.D=0.25; //0.25 //diffusion (u)
    S.Dm=S.D;
    S.alpha=26; //gravity (grad(h) in NS
    S.nu=0.15; //0.15; //inverse Reynolds number (Laplacian v)
    S.xi=4.0*omega*S.rho0;  //non-linear (~rho u^3) coeff in GL
    S.delta=0.0; //5th order term in u-equation (~rho u^5)
    lcoeff=0.5*5.5/(S.h0*S.h0);  // linear decay in NS
    S.beta=1.45*1.5*0.55*S.rho0/(S.h0*S.h0); //~ \rho/h^2 u in NS
    S.gamma=4.0*omega*S.rho0; //~ \rho u in GL
    S.Dr=1.0; //rotation diffusion
    S.eta=0.25*0.2*6.*omega*S.rho0/S.h0; //2.0; //~ -rho/h v in GL // 0.25*0.2*6.0*omega... in the figure
    S.kappa=.05*.05*S.rho0;//.05*.05*rho0; //~ -grad(rho) in GL // 0.01*0.05/rho0 in the figure
    mp=mp2=0.0;   // magnetic pressure
    mpon=mpon2=0;
    omega2on=-1;
    omega2=omega;
    omega2off=-1;
    omega_org=omega;
    
    S.hp=-50.0;
    S.ap=0.0; //0.04;
    S.p0=-0.1;

    //CPU arrays
    u=NULL;
    v=NULL;
    hrho=NULL;
    mu=NULL;
    tmpr=NULL;
    tp=NULL;

    //GPU data
    Gdata.dev=0; //GPUID (will be overridden by the GPU manager if available)
    Gdata.FFTinplace=false;

    Gdata.v=NULL;
    Gdata.u=NULL;
    Gdata.hrho=NULL;
    Gdata.mu=NULL;
    Gdata.tp=NULL;
    Gdata.FFTkern_hroh=NULL;
    Gdata.FFTkern_u=NULL;
    Gdata.FFTkern_v=NULL;
    Gdata.FFTkern_mu=NULL;
    Gdata.tmp1=NULL;
    Gdata.tmp2=NULL;
    Gdata.tmp3=NULL;
    Gdata.tmp4=NULL;
    
    //output control
    outctrl.format=FXDMF;
    outctrl.partden=true;
    outctrl.partv=true;
    outctrl.partvort=true;
    outctrl.fluidv=false;
    outctrl.mu=false;
    outctrl.h=false;
    outctrl.wellCF=false;
    outctrl.Tskip=0;
    outctrl.Tint=2500/2*1;
    outctrl.prefix="rho0p6_h2_";
    outctrl.path="OUT/";
    outctrl.gout=-1;
};

//---------------------------------------------------------------------------


//decontructor: clean up and release memory
CSpinners::~CSpinners()
{
    //Destroy FFT Plan & DEVICE arrays
    cufftDestroy(Gdata.fftPlan);

    if(Gdata.v!=NULL) cudaFree(Gdata.v);
    if(Gdata.u!=NULL) cudaFree(Gdata.u);
    if(Gdata.hrho!=NULL) cudaFree(Gdata.hrho);
    if(Gdata.mu!=NULL) cudaFree(Gdata.mu);
    if(Gdata.tp!=NULL) cudaFree(Gdata.tp);
    if(Gdata.FFTkern_hroh!=NULL) cudaFree(Gdata.FFTkern_hroh);
    if(Gdata.FFTkern_u!=NULL) cudaFree(Gdata.FFTkern_u);
    if(Gdata.FFTkern_v!=NULL) cudaFree(Gdata.FFTkern_v);
    if(Gdata.FFTkern_mu!=NULL) cudaFree(Gdata.FFTkern_mu);
    if(Gdata.tmp1!=NULL) cudaFree(Gdata.tmp1);
    if(Gdata.tmp2!=NULL) cudaFree(Gdata.tmp2);
    if(Gdata.tmp3!=NULL) cudaFree(Gdata.tmp3);
    if(Gdata.tmp4!=NULL) cudaFree(Gdata.tmp4);

    if(u!=NULL) delete[] u;
    if(v!=NULL) delete[] v;
    if(tp!=NULL) delete[] tp;
    if(hrho!=NULL) delete[] hrho;
    if(tmpr!=NULL) delete[] tmpr;
    if(mu!=NULL) delete[] mu;
}

//---------------------------------------------------------------------------

int CSpinners::initparams(int n, char* argv[])
{//overwrite deault built-in parameters set in constructor
    int a,i,j,k;
    REAL x;
    paramfilereader *pf=new paramfilereader();
    a=0;

    if(n>1) a=pf->cmdlineopenread(n,argv);
    if(a<1) a=pf->openread("conroll.ini");
    if(a>0)
    {
        ctype=pf->getint("ctype",ctype);
        Gdata.dev=pf->getint("GPUID",Gdata.dev);
        
        RS=pf->getint("RS",RS);
        
        initconf.type=pf->getint("init.config",initconf.type);
        initconf.p1=pf->getdouble("init.p1",initconf.p1);
        initconf.p2=pf->getdouble("init.p2",initconf.p2);
        initconf.p3=pf->getdouble("init.p3",initconf.p3);
        initconf.p4=pf->getdouble("init.p4",initconf.p4);
        
        conf.type=pf->getint("conforce.type",conf.type);
        conf.p1=pf->getdouble("conforce.p1",conf.p1);
        conf.p2=pf->getdouble("conforce.p2",conf.p2);
        conf.p3=pf->getdouble("conforce.p3",conf.p3);
        conf.p4=pf->getdouble("conforce.p4",conf.p4);
        conf.p5=pf->getdouble("conforce.p5",conf.p5);
        conf.N1=pf->getint("conforce.N1",conf.N1);
        conf.N2=pf->getint("conforce.N2",conf.N2);
        conf.N3=pf->getint("conforce.N3",conf.N3);
        conf.N4=pf->getint("conforce.N4",conf.N4);
        
        NTr=pf->getint("tracer.N",NTr);

        Nx=pf->getint("Nx",Nx);
        Ny=pf->getint("Ny",Ny);
        Lx=pf->getdouble("Lx",Lx);
        Ly=pf->getdouble("Ly",Ly);

        Nt=pf->getint("Nt",Nt);
        ht=pf->getdouble("ht",ht);
        
        
        noise.type=pf->getint("noise.type",noise.type);
        noise.step=pf->getint("noise.step",noise.step);
        noise.RS=pf->getint("noise.RS",RS); //uses RS if not specified
        noise.np1=pf->getdouble("noise.p1",noise.np1);
        noise.np2=pf->getdouble("noise.p2",noise.np2);
        
        

        S.alpha=pf->getdouble("alpha",S.alpha);
        S.nu=pf->getdouble("nu",S.nu);
        
        omega=pf->getdouble("omega",omega);
        omega2=pf->getdouble("omega2",omega2);
        omega2on=pf->getint("omega2on",omega2on);
        omega2off=pf->getint("omega2off",omega2off);
        
        S.De=pf->getdouble("De",S.De);
        S.D=pf->getdouble("D",S.D);
        S.Dr=pf->getdouble("Dr",S.Dr);
        
        
        S.h0=pf->getdouble("h0",S.h0);
        S.rho0=pf->getdouble("rho0",S.rho0);

        S.hp=pf->getdouble("pressure.hc",S.hp);
        S.ap=pf->getdouble("pressure.attr",S.ap);
        S.p0=pf->getdouble("pressure.p0",S.p0);
        mp=pf->getdouble("pressure.mag",mp);
        mpon=pf->getint("pressure.mag.on",mpon);
        mp2=pf->getdouble("pressure.mag2",mp2);
        mpon2=pf->getint("pressure.mag2.on",mpon2);
        
        //dependent parameters, can be manually overidden
        S.Dm=pf->getdouble("Dm",S.D);
        S.xi=pf->getdouble("xi",4.0*omega*S.rho0);
        S.delta=pf->getdouble("delta",0.0);
        S.beta=pf->getdouble("beta",1.45*1.5*0.55*S.rho0/(S.h0*S.h0));
        S.gamma=pf->getdouble("gamma",4.0*omega*S.rho0);
        S.eta=pf->getdouble("eta",0.25*0.2*6.0*omega*S.rho0/S.h0);
        S.kappa=pf->getdouble("kappa",0.05*0.05*S.rho0);
        

        if(pf->getstring("FFTinplace")=="yes") Gdata.FFTinplace=true;

        //only changes from default values
        if(pf->getstring("output.part.density")=="no") outctrl.partden=false;
        if(pf->getstring("output.part.velocity")=="no") outctrl.partv=false;
        if(pf->getstring("output.part.vorticity")=="no") outctrl.partvort=false;
        
        if(pf->getstring("output.fluid.velocity")=="yes") outctrl.fluidv=true;
        if(pf->getstring("output.part.mu")=="yes") outctrl.mu=true;
        if(pf->getstring("output.fluid.h")=="yes") outctrl.h=true;
        
        if(pf->getstring("output.well.CF")=="yes") outctrl.wellCF=true;
        
        
        outctrl.Tskip=pf->getint("output.timeskip",outctrl.Tskip);
        outctrl.Tint=pf->getint("output.interval",outctrl.Tint);
        outctrl.path=pf->getstring("output.path");
        outctrl.prefix=pf->getstring("output.prefix");
        if(outctrl.prefix=="") outctrl.prefix="conroll";
    }
    delete pf;

    srand(RS);                 // initialize random number generator

    if(a==0) //display help
    {
        printf("****** Usage information ******\n\n");
        printf(" %s <parameter file>\n",argv[0]);
        printf(" %s [parameter list: x=<value> y=<value> ...]\n",argv[0]);
        printf("\nparameter file contains paramters, one per line in format: x=<value>\n");


        return -1;
    }


    //dependent variable initialization
    N=Nx*Ny;

    hx=Lx/Nx; //for periodic BC, otherwise Lx/(Nx-1) ...
    hy=Ly/Ny;

    dkx=TWOPI/Lx;
    dky=TWOPI/Ly;
    
    if(noise.step<2) noise.step=1; //interval for noise "spikes"
    if(noise.type==1)
        noise.np1=sqrt(noise.np1*noise.step*ht/(hx*hy)); //rescale noise amplitude to be independent of discretization, includes ht from integration

    // block sizes for GPU

    Gdata.memC=N*sizeof(COMPLEX); //same as Vector2D
    Gdata.memR=N*sizeof(REAL);
    Gdata.memI=N*sizeof(int);

    Gdata.SBLOCK=dim3(MAXT,1);
    i=(N+Gdata.SBLOCK.x-1)/Gdata.SBLOCK.x;
    if(i>512) i=512;
    k=i*Gdata.SBLOCK.x;
    j=(N+k-1)/k; //y blocks
    Gdata.SGRID=dim3(i,j);

    if(NTr>0) {
        Gdata.PBLOCK=dim3(MAXT,1);
        i=(NTr+Gdata.PBLOCK.x-1)/Gdata.PBLOCK.x;
        if(i>512) i=512;
        k=i*Gdata.PBLOCK.x;
        j=(NTr+k-1)/k; //y blocks
        Gdata.PGRID=dim3(i,j);
    }
    
    //rescale the confinement force to its max range, i.e., p1 is the max force at the edge of its range
    if((conf.type==10) || (conf.type==11) || (conf.type==15) || (conf.type==16)) {
        x=conf.p3-conf.p2;
        if(x>EPS) conf.p1=conf.p1/x;
    } else if(conf.type==50) {
        x=conf.p5-conf.p2;
        if(x>EPS) conf.p1=conf.p1/x;
    }
    

    // Print out parameters of the system
    printf("Parameters of simulations:\n");
    printf(" - system size: %le x %le (%d x %d)\n",Lx,Ly,Nx,Ny);
    printf(" - simulation time: steps=%d, ht=%le\n",Nt,ht);
    printf(" - initial: type=%d (N=%d, %d, %d, %d); (p=%le, %le, %le, %le)\n",initconf.type,initconf.N1,initconf.N2,initconf.N3,initconf.N4,initconf.p1,initconf.p2,initconf.p3,initconf.p4);
    printf(" - confinement: type=%d (N=%d, %d, %d, %d); (p=%le, %le, %le, %le)\n",conf.type,conf.N1,conf.N2,conf.N3,conf.N4,conf.p1,conf.p2,conf.p3,conf.p4);
    
    

    return 0;
};

//---------------------------------------------------------------------------

int CSpinners::initialize()
{// Declarations
    cudaError_t err;
    size_t fmem,tmem;
    int i,j,idx;
    REAL r,d1,d2,a,x,y,v1,v2,phi,av,dx1,dx2,dy1,dy2,cx1,cx2,cy1,cy2;
    Vector2D vel;


    //GPU init

    printf("using CUDA device %d\n",Gdata.dev);
    //cudaThreadExit();
    cudaSetDevice(Gdata.dev);
    cudaMemGetInfo(&fmem,&tmem);
    printf("GPU memory before allocation free: %lu, total: %lu\n",fmem,tmem);

    // Allocating necessary arrays on GPU

    cudaMalloc((void**)&Gdata.v,Gdata.memC);
    cudaMalloc((void**)&Gdata.u,Gdata.memC);
    cudaMalloc((void**)&Gdata.hrho,Gdata.memC);
    cudaMalloc((void**)&Gdata.mu,Gdata.memC);
    cudaMalloc((void**)&Gdata.tmp1,Gdata.memC);
    cudaMalloc((void**)&Gdata.tmp2,Gdata.memC);
    cudaMalloc((void**)&Gdata.tmp3,Gdata.memC);
    cudaMalloc((void**)&Gdata.tmp4,Gdata.memC);

    cudaMalloc((void**)&Gdata.FFTkern_hroh,Gdata.memR);
    cudaMalloc((void**)&Gdata.FFTkern_u,Gdata.memR);
    cudaMalloc((void**)&Gdata.FFTkern_v,Gdata.memR);
    cudaMalloc((void**)&Gdata.FFTkern_mu,Gdata.memR);
    
    if(NTr>0) cudaMalloc((void**)&Gdata.tp,NTr*sizeof(Vector2D));

    // Create CUDA FFT plan
#ifdef __DBLPREC__
    cufftPlan2d(&Gdata.fftPlan, Nx, Ny, CUFFT_Z2Z);
#else
    cufftPlan2d(&Gdata.fftPlan, Nx, Ny, CUFFT_C2C);
#endif

    err=cudaGetLastError();
    if(err!=cudaSuccess)
        printf("CUDA error [%d] (alloc) : %s\n",err,cudaGetErrorString(err));

    cudaMemGetInfo(&fmem,&tmem);
    printf("GPU memory after allocation free: %lu, total: %lu\n",fmem,tmem);


    //CPU Array allocation
    u   = new Vector2D[N];
    v   = new Vector2D[N];
    hrho   = new Vector2D[N];
    mu   = new Vector2D[N];

    tmpr   = new REAL[N];
    
    if(NTr>0) tp   = new Vector2D[NTr];

    // Initializing CPU arrays randomly - default iconfig=0
    for(i=0; i<N; i++)
    {
        u[i].x = 0.02*(ran0()-0.5);
        u[i].y = 0.02*(ran0()-0.5);
        v[i].x = 0.0;
        v[i].y = 0.0;
        hrho[i].x = 1.0;
        hrho[i].y = 1.0;  //(i%Nx>0.15*Nx && i%Nx<0.85*Nx) && (i/Nx>0.15*Ny && i/Nx<0.85*Ny) ? 1.0 : 0.0001;
        mu[i].x= 0.02*(ran0()-0.5);
        mu[i].y= 0.02*(ran0()-0.5);
    }
    
    if(initconf.type==1) {
        for(i=0; i<N; i++)
        {
            u[i].x = 0.2*(ran0()-0.5);
            u[i].y = 0.2*(ran0()-0.5);
            v[i].x = 0.0;
            v[i].y = 0.0;
            hrho[i].x = 1.0;
            hrho[i].y = ran0()+0.5;  //(i%Nx>0.15*Nx && i%Nx<0.85*Nx) && (i/Nx>0.15*Ny && i/Nx<0.85*Ny) ? 1.0 : 0.0001;
            mu[i].x= 0.2*(ran0()-0.5);
            mu[i].y= 0.2*(ran0()-0.5);
        }
    }
    
    for(i=0; i<NTr; i++) {
        tp[i].x =Lx*ran0();
        tp[i].y =Ly*ran0();
    }
    
    
    if(initconf.type==102) { //place 2 vortices with same chirality on diagonal
        cx1=0.1*Lx;
        cy1=0.1*Ly;
        cx2=-cx1;
        cy2=-cy1;
        r=initconf.p1;
        a=1/r/r;
        av=0.0;
        for(idx=0; idx<N; idx++) { //override default
            i=idx%Nx;x=i*hx-0.5*Lx;
            j=idx/Nx;y=j*hy-0.5*Ly;
            dx1=x-cx1;dx2=x-cx2;
            dy1=y-cy1;dy2=y-cy2;
            d1=a*(dx1*dx1+dy1*dy1);
            d2=a*(dx2*dx2+dy2*dy2);
            v1=exp(-2.0*d1);
            v2=exp(-2.0*d2);
            av+=0.1+v1+v2;
            hrho[idx].y=0.1+v1+v2;
            
            vel.x=vel.y=0.0;
            if(d1>1e-6) {
                phi=atan2(dy1,dx1);
                vel.x+=-0.9*v1*sin(phi);  //counter clockwise
                vel.y+=0.9*v1*cos(phi);
            }
            if(d2>1e-6) {
                phi=atan2(dy2,dx2);
                vel.x+=0.9*v2*sin(phi);  // clockwise
                vel.y+=-0.9*v2*cos(phi); //
            }
            u[idx].x=v[idx].x=vel.x;
            u[idx].y=v[idx].y=vel.y;
        }
        av=(1.0*N)/av;
        for(idx=0; idx<N; idx++) hrho[idx].y=av*hrho[idx].y;
    }

    //no need to initialize tmpr, Gdata.tmpX

    cudaMemcpy(Gdata.u, u, Gdata.memC,cudaMemcpyHostToDevice); // Copy u to Device memory
    cudaMemcpy(Gdata.v, v, Gdata.memC,cudaMemcpyHostToDevice);
    cudaMemcpy(Gdata.hrho, hrho, Gdata.memC,cudaMemcpyHostToDevice);
    cudaMemcpy(Gdata.mu, mu, Gdata.memC,cudaMemcpyHostToDevice);
    
    if(NTr>0) cudaMemcpy(Gdata.tp, tp, NTr*sizeof(Vector2D),cudaMemcpyHostToDevice);

    err=cudaGetLastError();
    if(err!=cudaSuccess)
    printf("CUDA error [%d] (mem copy): %s\n",err,cudaGetErrorString(err));

    //calculate FFT kernels
    G_init_FFTkern_2D  <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,ht,dkx,dky,S.De,0.0,Gdata.FFTkern_hroh);
    G_init_FFTkern_2D  <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,ht,dkx,dky,S.nu,lcoeff,Gdata.FFTkern_v);
    G_init_FFTkern_2D  <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,ht,dkx,dky,S.D,0.0,Gdata.FFTkern_u);
    G_init_FFTkern_2D  <<< Gdata.SGRID, Gdata.SBLOCK >>> (Nx,Ny,ht,dkx,dky,S.Dm,0.0,Gdata.FFTkern_mu);


    err=cudaGetLastError();
    if(err!=cudaSuccess)
        printf("CUDA error [%d] (FFT kern init): %s\n",err,cudaGetErrorString(err));

    return 0;
};

//====================================================================================================

bool CSpinners::cerr(const char *s)
{
    cudaError_t err=cudaGetLastError();
    if(err==cudaSuccess)
        return false;
    printf("CUDA error [%s]: %s",s,cudaGetErrorString(err));
    return true;
};


//====================================================================================================
//====================================================================================================

int main(int argc, char* argv[])
{   CSpinners *cr;
    int res;

    printf("GPU-enabled continuum roller simulations, version %d.%d.%d (",((Sversion>>16)&0xFF),((Sversion>>8)&0xFF),((Sversion)&0xFF));
#ifdef __DBLPREC__
    printf("double precision");
#else
    printf("single precision");
#endif
    printf(")\n");


    cr=new CSpinners();
    if((res=cr->initparams(argc,argv))!=0)
    {
        printf("program stopped, parameter initialization error %d\n",res);
        return 0;
    }
    printf("params set, initializing ... ");
    if((res=cr->initialize())!=0)
    {
        printf("program stopped, initialization error %d\n",res);
        return 0;
    }
    printf("done\nstarting the simulation\n");
    cr->run();
    printf("finished - cleaning up\n");
    delete cr;


    return 0;
};
//====================================================================================================
