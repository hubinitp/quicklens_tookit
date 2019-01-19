#/usr/bin/env python
# --
# quicklens/examples/lens/make_lensing_estimators.py
# --
# generates a set of lensed maps in the flat-sky limit, then runs
# quadratic lensing estimators on them to estimate phi. plots the
# auto-spectrum of the phi estimates as well as their cross-spectra
# with the input phi realization, and a semi-analytical estimate
# of the noise bias.

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import quicklens as ql

# simulation parameters.
nsims      = 1 #25 bh
lmin       = 10
lmax       = 800
nx         = 512 # number of pixels.
dx         = 1./60./180.*np.pi # pixel width in radians.

#ny         = 512 # number of pixels. bh
ny         = 256 # number of pixels. bh
dy         = 1./60./180.*np.pi # pixel width in radians. bh

nlev_t     = 5.  # temperature noise level, in uK.arcmin.
nlev_p     = 5.  # polarization noise level, in uK.arcmin.
bl         = ql.spec.bl(1., lmax) # beam transfer function.
#bh: bl returns the map-level transfer function for a symmetric Gaussian beam.

pix        = ql.maps.pix(nx,dx)
#pix        = ql.maps.pix(nx,dx,ny,dy) #bh

# analysis parameters
estimators = [ ('ptt', 'r'), ('pee', 'g'), ('peb', 'b') ] # (estimator, plotting color color) pairs ('ptt' = TT estimator, etc.)

mask       = np.ones( (nx, nx) ) # mask to apply when inverse-variance filtering.
#mask       = np.ones( (ny, nx) ) # mask to apply when inverse-variance filtering. bh
                                 # currently, no masking.
                                 # alternatively, cosine masking:
                                 # x, y = np.meshgrid( np.arange(0,nx), np.arange(0,nx) )
                                 # mask = np.sin( np.pi/nx*x )*np.sin( np.pi/nx*y )

mc_sims_mf = None                # indices of simulations to use for estimating a mean-field.
                                 # currently, no mean-field subtraction.
                                 # alternatively: np.arange(nsims, 2*nsims)

npad       = 1

# plotting parameters.
t          = lambda l: (l+0.5)**4/(2.*np.pi) # scaling to apply to cl_phiphi when plotting.
lbins      = np.linspace(10, lmax, 30)       # multipole bins.

# cosmology parameters.
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)
clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx)).get_ml(lbins, t=t)
#clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx,ny=ny,dy=dy)).get_ml(lbins, t=t) #bh

# make libraries for simulated skies.
sky_lib    = ql.sims.cmb.library_flat_lensed(pix, cl_unl, "temp/sky")
#bh: sky_lib store the simulated lensed cmb signal, generattor of random seeds
obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp/obs")
#bh: obs_lib stroe the beam + noise added map, generator of random seeds

#print dir(sky_lib.phase)
#print sky_lib.__dict__
#print sky_lib.phase.random.__doc__
#print dir(obs_lib.phase)
#quit()

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
#bh: cl here is the theoretical one, without any noise, for each ell, clmat is a 3by3 matrix 
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
#bh: transf here is the tebfft modes from beam bl
#def cl2tebfft(cl, pix):""" returns a maps.tebfft object with the pixelization pix and [T,E,B]FFT(lx,ly) = linear interpolation of cl.cltt[l], cl.clee[l], cl.clbb[l] at l = sqrt(lx**2 + ly**2). """

#print dir(transf) #bh
#print dir(transf.inverse().inverse())
'''
print dir(cl)
print cl.__dict__
print cl.clmat.shape
'''
'''
print dir(transf)
print transf.__dict__
print transf.bfft.shape
print transf.efft.shape
print transf.tfft.shape
quit()
'''

'''
for i in range(0,cl.lmax): #bh
	op_data = open('cls.dat','a')
	np.savetxt(op_data, np.c_[i,cl.clmat[i,0,0],cl.clmat[i,1,1],cl.clmat[i,2,2]])
	op_data.close()
'''

ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )
#bh: if rectangle is needed: library_l_mask: def __init__(self, ivf_lib, lmin=None, lxmin=None, lxmax=None, lmax=None, lymin=None, lymax=None)
#bh: ivf_lib give, a tebfft matrix (fft map), each element of the matrix (fft map) has its value equals to F_l, where F_l is the Wiener filter. 
#bh: ivf: Fl = 1/(Cl+Nl) 
#bh: ivf_lib maintains: nl, the noise tebfft map, tl, the beam^-1 tebfft map, fl, the ivf tebfft map, transf, the beam tebfft map 
#bh: ivf_lib encapsulates all the ingrediants of ivf tebfft map, but the realization has not yet produced

'''
bh_test = ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask)
print dir(bh_test.obs_lib.phase)
print bh_test.obs_lib.phase.__dict__
quit()
'''

qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp/qest", npad=npad)
#bh: qest_lib shall hold the estimated phi map
'''
print dir(qest_lib)
print qest_lib.__dict__
quit()
'''

qest_lib_kappa = ql.sims.qest.library_kappa(qest_lib, sky_lib)

qecl_lib = ql.sims.qecl.library(qest_lib, lib_dir="temp/qecl", mc_sims_mf=mc_sims_mf, npad=npad)
#bh: qecl_lib shall hold the estimated spectrum
qecl_kappa_cross_lib = ql.sims.qecl.library(qest_lib, qeB=qest_lib_kappa, lib_dir="temp/qecl_kappa", npad=npad)

# --
# run estimators, make plots.
# --

pl.figure()

p = pl.semilogy

cl_unl.plot('clpp', t=t, color='k', p=p)

for key, color in estimators:
    qr = qest_lib.get_qr(key) #bh: qr is the repsonse function, stored as a fft map
    #bh: response function RL=AL^-1 for the Hu-Okamoto flat-sky quadratic estimator
    #bh: qr is deterministic from the input Cl, not random realization
    #bh: return qe.fill_resp( qs, ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy), f1.fft, f2.fft, npad=self.npad) 

    qcr = qecl_lib.get_qcr_lcl(key) #bh: specs.lcl class, holds 1d spectrum
    #bh: get_qcr holds the covariance of the qeA x qeB, which equals to qr(A) x qr(B), algebraic multipication
    #bh: get_qcr is the analytical covariance of qeA x qeB
    #bh: get_qcr_lcl transform the 2D covariance into 1D covariance 

    ''' 
    print qr.__doc__   
    print dir(qr)
    print qr.__dict__
    print qr.fft.shape
    raw_input()
    '''

    '''
    # intialize averagers.
    cl_phi_x_phi_avg    = ql.util.avg()
    cl_phi_x_est_avg    = ql.util.avg()
    cl_est_x_est_avg    = ql.util.avg()
    cl_est_x_est_n0_avg = ql.util.avg()
    '''

    if (key == 'ptt'): 
	print 'in ptt'
    	bh_n0_ptt = (qecl_lib.get_sim_ncl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh 
    	bh_est_ptt = (qecl_lib.get_sim_qcl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh
    if (key == 'pee'): 
	print 'in pee'
    	bh_n0_pee = (qecl_lib.get_sim_ncl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh 
    	bh_est_pee = (qecl_lib.get_sim_qcl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh
    if (key == 'peb'): 
	print 'in peb'
    	bh_n0_peb = (qecl_lib.get_sim_ncl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh 
    	bh_est_peb = (qecl_lib.get_sim_qcl_lcl(key, 0) / qcr).get_ml(lbins, t=t) #bh

    '''
    print bh_n0.ls
    print bh_est.ls
    raw_input()
    '''

    #np.savetxt('./bh_try.dat', np.c_[bh_n0.ls,bh_est.specs['cl']]) #bh
    #np.savetxt('./bh_try.dat', np.c_[bh_n0.ls,bh_n0.specs['cl'].real]) #bh

    '''
    # average power spectrum estimates.
    for idx, i in ql.util.enumerate_progress(np.arange(0, nsims), label="averaging cls for key=%s"%key):        
	#cl_est_x_est_n0_avg.add( (qecl_lib.get_sim_ncl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
	bh_n0 = (qecl_lib.get_sim_ncl_lcl(key, i) / qcr).get_ml(lbins, t=t) #bh 
	#bh: get_ml rebins the ell modes to get the band power
	#bh: qcr is R(qeA) x R(qeB) to normalize the estimated spectrum 

	
	#bh_test = qecl_lib.get_sim_ncl_lcl(key, i)
	#print dir(bh_test)
	#print bh_test.__dict__
	#raw_input()
	
        #cl_est_x_est_avg.add( (qecl_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
        bh_est = (qecl_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) #bh 
        
	#bh: mostly important line! 
	#bh: get_sim_qft(kA, i) get the estimation of phi via kA estimator from the simulation i
	#bh: get_sim_qcl gives the maps.cfft object containing the cross-power between two quadratic estimators (qeA x qeB) from simulation i, here the mean field subtraction is operated 
	#bh: get_sim_qcl_lcl output the spectrum from the get_sim_qcl
	#bh: qcr = RL(qeA) x RL(qeB), gives the normalization

        #cl_phi_x_est_avg.add( (qecl_kappa_cross_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
        bh_kappa = (qecl_kappa_cross_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) #bh
    '''

    # plot lensing spectra.
    #(1./qr).get_ml(lbins, t=t).plot(color='y')            # analytical n0 bias.
    #bh: for Hu-Okamoto flat-sky quadratic estimator, N0 = AL = RL^-1
    #bh: maps.tebfft.get_ml returns a Cl from the given tebfft map (Cl = 1/norm * sum a_lm)
     
    #cl_phi_x_est_avg.plot(p=p, color=color)               # lensing estimate x input lensing. 
    #cl_est_x_est_avg.plot(p=p, color=color, lw=3)         # lensing estimate auto spectrum. 
    #(bh_est_ptt-bh_n0_ptt).plot(p=p, color=color, lw=3)         # lensing estimate auto spectrum.
    #cl_est_x_est_n0_avg.plot(p=p, color=color, ls='--')   # semi-analytical n0 bias.

    #(clpp + cl_est_x_est_n0_avg).plot(color='k', ls=':')  # theory lensing power spectrum + semi-analytical n0 bias.

    
np.savetxt('./n0.dat', np.c_[bh_n0_ptt.ls,bh_n0_ptt.specs['cl'].real,bh_n0_pee.specs['cl'].real,bh_n0_peb.specs['cl'].real]) #bh
np.savetxt('./est.dat', np.c_[bh_est_ptt.ls,bh_est_ptt.specs['cl'].real,bh_est_pee.specs['cl'].real,bh_est_peb.specs['cl'].real]) #bh

'''
pl.xlabel(r'$l$')
pl.ylabel(r'$(l+\frac{1}{2})^4 C_l^{\phi\phi} / 2\pi$')

pl.xlim(lbins[0], lbins[-1])


for key, color in estimators:
    pl.plot( [-1,-2], [-1,-2], label=r'$\phi^{' + key[1:].upper() + r'}$', lw=2, color=color )
pl.legend(loc='lower left')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)
    
pl.ion()
pl.show()

pl.savefig('estimator.png')
'''

