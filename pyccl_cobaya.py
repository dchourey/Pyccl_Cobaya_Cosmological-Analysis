#This runs the MCMC sampler using cobaya to run with following situations:
#Using data vector D from data file "data_vector_l_cl" at z = 1.0537. The data vector will be in the form of angular shear power spectra.
#The model will be used with the Boltzmann CLASS transfer function and Halofit non-linear matter power spectra
#proposal scale = 0.2


#Importing required packages

import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import classy
import cobaya
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
import camb
from astropy.io import fits


#Importing angular power spectrum and corrosponding multipole values

data = np.genfromtxt('data_vector_l_cl', dtype = float, delimiter='     ')
ell = data[:, 0]
data_11 = 2*np.pi*data[:, 1]

#Covariance

hdul = fits.open('cov_er_oss_media.fits')  # open a FITS file
cov = hdul[0].data

icov = np.linalg.inv(cov) #inverse of cov


#Specify the redshift value
z_s = 1.0537



#defining likelihood as a normal distribution and also the derived paramert omega_m and sigma_8

def lnprob(om_c, om_b, ho, As, ns, nu_eff):

	#Definging the model cosmology

	cosmo = ccl.Cosmology(Omega_c= om_c, Omega_b=om_b, h=ho, A_s=As*10**(-9), n_s=ns, Neff=nu_eff, 
		              transfer_function='boltzmann_class',  matter_power_spectrum='halofit',
		              mass_function='shethtormen',
		              halo_concentration='duffy2008'
		             )
	#creating the tracer 
	tracer_1 = ccl.CMBLensingTracer(cosmo, z_source=z_s)
	
	#calculating the angular power spectrum using above tracers

	model_11 = ccl.angular_cl(cosmo, tracer_1, tracer_1, ell)
	
	#Definging the (data-model) 

	#diff_11 = np.subtract( data_11, model_11)
	diff_11  = data_11 - model_11
	#print( model_11)
	
	#Definging the the gaussian likelihood

	lnprob_11 = (-1)*np.dot(diff_11, np.dot(icov, diff_11))/2.0
	
	
	#Estimting the model halo matter power spectrum to get matter power spectrum normalization Sigma_8, this saves Sigma_8 in cache which is recovered using ccl.sigma8 below.

	kmin, kmax, nk = 1e-4, 1e1, 256
	k = np.logspace(np.log10(kmin), np.log10(kmax), nk) #Defining the wavenumber k
	a = 1. / (1. + z_s)  # Scale factor a z=0 

	wk_model = ccl.halomodel_matter_power(cosmo, k, a)

	totalprob = lnprob_11
	sig8 = ccl.sigma8(cosmo)
	om_m = om_c + om_b
	S8 = sig8 * ((om_m / 0.3)**0.5)
	derived = {"om_m": om_m, "sig8": sig8, "S8": S8}
	return totalprob, derived


#Defining information for cobaya input

info = {
    'params': {
        # Sampled cosmological parameter
              
         'om_c': {'prior': { 'min': 0.1, 'max': 0.9},  'ref': {'dist': 'norm', 'loc': 0.5, 'scale': .1}, 'latex': '\Omega_c'},
        'om_b': {'prior': { 'min': 0.03, 'max': 0.07},  'ref': {'dist': 'norm', 'loc': 0.05, 'scale': .004}, 'latex': '\Omega_b'},
        'ho': {'prior': { 'min': 0.55, 'max': 0.91},  'ref': {'dist': 'norm', 'loc': 0.7, 'scale': 0.04}, 'latex': 'h'},
        'As': {'prior': { 'min': .5, 'max': 5.},  'ref': {'dist': 'norm', 'loc': 2., 'scale': .5}, 'latex': 'A_\\mathrm{s} (10^{-9})'},
        'ns': {'prior': { 'min': 0.87, 'max': 1.07},  'ref': {'dist': 'norm', 'loc': 0.9, 'scale': 0.03}, 'latex': 'n_\\mathrm{s}'},
        'nu_eff': {'prior': { 'min': 1., 'max': 5.},  'ref': {'dist': 'norm', 'loc': 3., 'scale': .3}, 'latex': 'N_{eff}'},
        
        
	#Derived parameters
	'om_m': {'latex': '\Omega_m'},
	'sig8': {'latex': '\sigma_8'},
	'S8': {'latex': '\\mathrm{s}_8'}},

	#Defining the gaussian likelihood

    'likelihood': {'my_cl_like': {
        "external": lnprob, 
        "output_params": ['om_m', 'sig8', 'S8']}},
        
        #definging the sampler properties
        
	 'sampler': {'mcmc': {'Rminus1_cl_stop': 0.2,
                      'Rminus1_stop': 0.2,
                      'covmat': 'auto',
                      'drag': False,
                      'oversample_power': 0.4,
                      'proposal_scale': 0.2}},
                      
      #setting the chains location
      	
    'output': 'chains/testrun'}



import sys
for k, v in {"-f": "force", "-r": "resume", "-d": "debug"}.items():
    if k in sys.argv:
        info[v] = True


#Running MCMC sampler

from cobaya.model import get_model

updated_info, sampler = run(info,  "-r")


