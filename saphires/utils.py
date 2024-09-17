'''
############################ SAPHIRES utils ###########################
                     Written by Ben Tofflemire, 2019
#######################################################################
This file is part of the SAPHIRES python package.

SAPHIRES is free software: you can redistribute it and/or modify it 
under the terms of the MIT license.

SAPHIRES is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the MIT license with SAPHIRES.  
If not, see <http://opensource.org/licenses/MIT>.

Module Description:
A collection of utility functions used in the SAPHIRES package. The 
only function in here you are likely to use is prepare. The rest are
called by other functions in the bf, xc, or io modules that are 
tailored for typical users.

Functions are listed in alphabetical order.
'''

# ---- Standard Library
import sys
import copy as copy
import os
from datetime import datetime
# ----

# ---- Third Party
import numpy as np

from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pickle as pkl

import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.time import Time
from astropy import constants as const

from barycorrpy import utc_tdb
from barycorrpy import get_BC_vel

import emcee

import corner
# ---- 

# ---- Project
from saphires.extras import bspline_acr as bspl
# ----

py_version = sys.version_info.major
if py_version == 3:
	nplts = 'U'	#the numpy letter for a string
	p_input = input
if py_version == 2:
	nplts = 'S' #the numpy letter for a string
	p_input = raw_input

np.seterr(invalid='ignore')

def air2vac(w_air):
	'''
	Air to vacuum conversion formula derived by N. Piskunov
	IAU standard:
	http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

	Parameters
	----------
	w_air : array-like
		Array of air wavelengths assumed to be in Angstroms

	Returns
	-------
	w_vac : array-like
		Array of vacuum wavelengths converted from w_air
	'''

	s = 10**4/w_air

	n = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 
	     0.0001599740894897 / (38.92568793293 - s**2))

	w_vac = w_air*n

	return w_vac


def apply_shift(t_f_names,t_spectra,rv_shift,shift_style='basic'):
	'''
	A function to apply a velocity shift to an input spectrum.

	The shift is made to the 'nflux' and 'nwave' arrays.

	The convention may be a bit wierd, think of it like this:
	If there is a feature at +40 km/s and you want that feature 
	to be at zero velocity, put in 40. 

	Whatever velocity you put in will be put to zero. 

	The shifted velocity is stored in the 'rv_shift' header 
	for each dictionary spectrum. Multiple shifts are stored
	i.e. shifting by 40, and then 40 again will result in a
	'rv_shift' value of 80. 

	Parameters
	----------
    t_f_names: array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum.

	rv_shift : float
		The velocity (in km/s) you want centered at zero.

	shift_style : str, optional
		Parameter defines how to shift is applied. Options are 'basic' and
		'inter'. The defaul is 'basic'.
		- The 'basic' option adjusts the wavelength asignments with the
		  standard RV shift. Pros, you don't interpolate the flux values; 
		  Cons, you change the wavelength spacing from being strictly linear. 
		  The Pros outweight the cons in most scenarios.
		- The 'inter' option leaves the wavelength gridpoints the same, but 
		  shifts the flux with an interpolation. Pros, you don't change the 
		  wavelength spacing; Cons, you interpolate the flux, before you 
		  interpolate it again in the prepare step. Interpolating flux is not
		  the best thing to do, so the less times you do it the better. 
		  The only case I can think of where this method would be better is
		  if your "order" spanned a huge wavelength range. 

	Returns
	-------
	spectra_out : python dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will be a copy of t_specrta, but with updates to the following keywords.

		['nwave']    - The shifted wavelength array
		['nflux']    - The shifted flux array
		['rv_shift'] - The value the spectrum was shifted in km/s

	'''
	c = const.c.to('km/s').value

	spectra_out = copy.deepcopy(t_spectra)

	for i in range(t_f_names.size):
		if shift_style == 'inter':
			w_unshifted = spectra_out[t_f_names[i]]['nwave']
		
			w_shifted = w_unshifted/(1-(-rv_shift/(c)))
			
			f_shifted_f = interpolate.interp1d(w_shifted,spectra_out[t_f_names[i]]['nflux'])
		
			shift_trim = ((w_unshifted>=np.min(w_shifted))&(w_unshifted<=np.max(w_shifted)))
		
			w_unshifted = w_unshifted[shift_trim]
		
			spectra_out[t_f_names[i]]['nwave'] = w_unshifted
		
			f_out=f_shifted_f(w_unshifted)
		
			spectra_out[t_f_names[i]]['nflux'] = f_out	

		if shift_style == 'basic':
			w_unshifted = spectra_out[t_f_names[i]]['nwave']
		
			w_shifted = w_unshifted/(1-(-rv_shift/(c)))

			spectra_out[t_f_names[i]]['nwave'] = w_shifted

		w_range = spectra_out[t_f_names[i]]['w_region']

		if w_range != '*':
			w_split = np.empty(0)
			w_rc1 = w_range.split('-')
			for j in range(len(w_rc1)):
				for k in range(len(w_rc1[j].split(','))):
					w_split = np.append(w_split,float(w_rc1[j].split(',')[k]))

			w_split_shift = w_split/(1-(-rv_shift/(c)))

			w_range_shift = ''
			for j in range(w_split_shift.size):
				if (j/2.0 % 1) == 0: #even
					w_range_shift = w_range_shift + str(np.round(w_split_shift[j],2))+'-'
				if (j/2.0 % 1) != 0: #odd
					w_range_shift = w_range_shift + str(np.round(w_split_shift[j],2))+','

			if w_range_shift[-1] == ',':
				w_range_shift = w_range_shift[:-1]
	
			spectra_out[t_f_names[i]]['w_region'] = w_range_shift

		spectra_out[t_f_names[i]]['rv_shift'] = spectra_out[t_f_names[i]]['rv_shift'] + rv_shift

	return spectra_out


def bf_map(template,m):
    '''
    Creates a two dimensional array from a template by shifting one unit
    of the wavelength (velocity) array m/2 times to the right and m/2 to 
    the left. This array, also known as the design matrix (des), is then 
    input to bf_solve.

    Template is the resampled flux array in log(lambda) space.

    Parameters
    ----------
    template : array-like
		The logarithmic wavelengthed spectral template. Must have an 
		even numbered length.

    m : int
    	The number of steps to shift the template. Must be odd

    Returns
    -------
    t : array-like
    	The design matrix 
    '''
    t=0
    n=template.size

    if (n % 2) != 0:
        print('Input array must be even')
        return

    if (m % 2) != 1:
        print('Input m must be odd')
        return

    t=np.zeros([n-m+1,m])
    for j in range(m):
        for i in range(m//2,n-m//2):
            t[i-m//2,j]=template[i-j+m//2]

    return t


def bf_singleplot(t_f_names,t_spectra,for_plotting,f_trim=20):
	'''
	A function to make a mega plot of all the spectra in a target 
	dictionary. 

	Parameters
	----------
	t_f_names : array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum. Output of one of the 
		saphires.io read-in functions.

	for_plotting : python dictionary
		A dictionary with all of the things you need to plot the fit profiles 
		from saphires.bf.analysis.

	f_trim : int
		The amount of points to trim from the edges of the BF before the fit is 
		made. The edges are usually noisey. Units are in wavelength spacings. 
		The default is 20.

	Returns
	-------
	None

	'''

	pp=PdfPages(t_f_names[0].split('[')[0].split('.')[0]+'_allplots.pdf')

	for i in range(t_f_names.size):
		w1=t_spectra[t_f_names[i]]['vwave']
		target=t_spectra[t_f_names[i]]['vflux']
		template=t_spectra[t_f_names[i]]['vflux_temp']
		vel=t_spectra[t_f_names[i]]['vel']
		temp_name=t_spectra[t_f_names[i]]['temp_name']
		bf_sols=t_spectra[t_f_names[i]]['bf']

		bf_smooth = for_plotting[t_f_names[i]][0]
		func = for_plotting[t_f_names[i]][1]
		gs_fit = for_plotting[t_f_names[i]][2]
	
		m=vel.size

		fig,ax=plt.subplots(2,sharex=True)
		ax[0].set_title('Template',fontsize=8)
		ax[0].set_ylabel('Normalized Flux')
		ax[0].plot(w1,template)
		ax[1].set_title('Target',fontsize=8)
		ax[1].set_ylabel('Normalized Flux')
		ax[1].plot(w1,target)
		ax[1].set_xlabel(r'$\rm{\lambda}$ ($\rm{\AA}$)')
		plt.tight_layout(pad=0.4)
		pp.savefig()
		plt.close()

		fig,ax=plt.subplots(1)
		ax.plot(vel,bf_smooth,color='lightgrey',lw=4,ls='-')
		ax.set_ylabel('Broadening Function')
		ax.set_xlabel('Radial Velocity (km/s)')

		if gs_fit.size == 10:
			ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		                                         gs_fit[0],gs_fit[1],
		                                         gs_fit[2],gs_fit[9]),
					lw=2,ls='--',color='b',
					label='Amp1: '+str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
			ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		                                         gs_fit[3],gs_fit[4],
		                                         gs_fit[5],gs_fit[9]),
					lw=2,ls='--',color='r',
					label='Amp2: '+str(np.round(gs_fit[3]*gs_fit[5]*np.sqrt(2.0*np.pi),3)))
			ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		                                         gs_fit[6],gs_fit[7],
		                                         gs_fit[8],gs_fit[9]),
					lw=2,ls='--',color='g',
					label='Amp3: '+str(np.round(gs_fit[6]*gs_fit[8]*np.sqrt(2.0*np.pi),3)))
			ax.legend()
			
		if gs_fit.size == 7:
			#if func == gauss_rot_off:
			#	ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    #	                                     gs_fit[0],gs_fit[1],
		    #	                                     gs_fit[2],gs_fit[6]),
			#			lw=2,ls='--',color='b',
			#			label='Amp1: '+str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
			#
			#	ax.plot(vel[f_trim:-f_trim],rot_pro(vel[f_trim:-f_trim],
		    #	                                     gs_fit[3],gs_fit[4],
		    #	                                     gs_fit[5],gs_fit[6]),
			#			lw=2,ls='--',color='r',
			#			label='Amp2: '+str(np.round(gs_fit[3],3)))

			if func == d_gaussian_off:
				ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    	                                     gs_fit[0],gs_fit[1],
		    	                                     gs_fit[2],gs_fit[6]),
						lw=2,ls='--',color='b',
						label='Amp1: '+str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
				
				ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    	                                     gs_fit[3],gs_fit[4],
		    	                                     gs_fit[5],gs_fit[6]),
						lw=2,ls='--',color='r',
						label='Amp2: '+str(np.round(gs_fit[3]*gs_fit[5]*np.sqrt(2.0*np.pi),3)))
			ax.legend()

		ax.plot(vel[f_trim:-f_trim],func(vel[f_trim:-f_trim],*gs_fit),
	        	lw=1,ls='-',color='k')
		
		plt.tight_layout(pad=0.4)
		pp.savefig()
		plt.close()

	pp.close()

	return


def bf_solve(des,u,ww,vt,target,m):
    '''
    Takes in the design matrix, the output of saphires.utils.bf_map, 
    and creates an array of broadening functions where each row is 
    for a different order of the solution.

    The last index out the output (m-1) is the full solution.

    All of them here for completeness, but in practice, the full 
    solution with gaussian smoothing is used to derive RVs and flux ratios.
    
    Parameters
    ----------
	des : arraly-like
		Design Matrix computed by saphires.utils.bf_map

	u : array-like
		One of the outputs of the design matrix's singular value 
		decomposition

	ww : array-like
		One of the outputs of the design matrix's singular value 
		decomposition

	vt : array-like
		One of the outputs of the design matrix's singular value 
		decomposition

	target : array-like
		The target spectrum the corresponds to the template spectrum 
		that was used to make the design matrix.

	m : int
		Number of pixel shifts to compute

	Returns
    -------
    b_sols : array-like
    	Matrix of BF solutions for different orders. The last order if the 
    	one to use. 

    sig : array-like
    	Uncertainty array for each order. 
    '''

    #turning ww into a matrix
    ww_mat=np.zeros([m,m])
    for i in range(m):
        ww_mat[i,i]=ww[i]

    #turning ww into its inverse (same as the transpose in this case) matrix.
    ww1=np.zeros([m,m])
    for i in range(m):
        ww1[i,i]=1.0/ww[i]
        
    #just making sure all the math went okay.
    if np.allclose(des, np.dot(np.dot(u,ww_mat),vt)) == False:
        print('Something went wrong with the matrix math in bf_sols')
        return

    #trimming target spectrum to have the right length
    target_trim=target[m//2:target.size-m//2]
    
    #computing the broadening function
    b_sols=np.zeros([m,m])
    for i in range(m):
        wk=np.zeros([m,m])
        wb=ww[0:i]
        for j in range(wb.size):
            wk[j,j]=1.0/wb[j]
        b_sols[i,:] = np.dot(np.dot(np.transpose(vt), wk),
                             np.dot(np.transpose(u),target_trim))
    
    #computing the error of the fit between the two. 
    sig=np.zeros(m)
    pred=np.dot(des,np.transpose(b_sols))
    for i in range(m):
        sig[i]=np.sqrt(np.sum((pred[:,i]-target_trim)**2)/m)
    
    return b_sols,sig


def bf_text_output(ofname,target,template,gs_fit,rchis,rv_weight,fit_int):
	'''
	A function to output the results of saphires.bf.analysis to a text file.
	
	Parameters
	----------
	ofname : str
		The name of the output file.

	target : str
		The name of the target spectrum.

	template : str
		The name of the template spectrum.

	gs_fit : array-like
		An array of the profile fit parameters from saphire.bf.analysis.

	rchis : float
		The reduced chi square of the saphires.bf.analysis fit with the data.

	fit_int : array-like
		Array of profile integrals from the saphires.bf.analysis fit.

	Returns
	-------
	None

	'''

	if os.path.exists('./'+ofname) == False:
		f=open(ofname,'w')
		f.write('#Column Details\n')
		f.write('#System Time\n')
		f.write('#Fit Parameters - For each profile fit, the following:')
		f.write('# - Amp, RV (km/s), Sigma, Integral\n')
		f.write('#Reduced Chi Squared of Fit\n')
		f.write('#Target File Name\n')
		f.write('#Template File Name\n')
	    
	else:
		f=open(ofname,'a')

	f.write(str(datetime.now())[0:-5]+'\t')

	if gs_fit.size==10:
		peak_ind=np.argsort([gs_fit[0],gs_fit[3],gs_fit[6]])[::-1]*3+1
		f.write(str(np.round(gs_fit[peak_ind[0]-1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[0]],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[0]+1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[0]-1]*
		                        gs_fit[peak_ind[0]+1]*np.sqrt(2*np.pi),2))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[1]-1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[1]],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[1]+1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[1]-1]*
		                        gs_fit[peak_ind[1]+1]*np.sqrt(2*np.pi),2))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[2]-1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[2]],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[2]+1],4))+'\t')
		f.write(str(np.round(gs_fit[peak_ind[2]-1]*
		                        gs_fit[peak_ind[2]+1]*np.sqrt(2*np.pi),2))+'\t')

	if gs_fit.size==7:
		if fit_int[0]>fit_int[1]:
			f.write(str(np.round(gs_fit[0],4))+'\t')
			f.write(str(np.round(gs_fit[1],4))+'\t')
			f.write(str(np.round(gs_fit[2],4))+'\t')
			f.write(str(np.round(fit_int[0],2))+'\t')
			f.write(str(np.round(gs_fit[3],4))+'\t')
			f.write(str(np.round(gs_fit[4],4))+'\t')
			f.write(str(np.round(gs_fit[5],4))+'\t')
			f.write(str(np.round(fit_int[1],2))+'\t')
		else:
			f.write(str(np.round(gs_fit[3],4))+'\t')
			f.write(str(np.round(gs_fit[4],4))+'\t')
			f.write(str(np.round(gs_fit[5],4))+'\t')
			f.write(str(np.round(fit_int[1],2))+'\t')
			f.write(str(np.round(gs_fit[0],4))+'\t')
			f.write(str(np.round(gs_fit[1],4))+'\t')
			f.write(str(np.round(gs_fit[2],4))+'\t')
			f.write(str(np.round(fit_int[0],2))+'\t')

	if gs_fit.size==4:
		f.write(str(np.round(gs_fit[0],4))+'\t')
		f.write(str(np.round(gs_fit[1],4))+'\t')
		f.write(str(np.round(gs_fit[2],4))+'\t')
		f.write(str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi),2))+'\t')
	
	f.write(str(np.round(rchis,3))+'\t')
	f.write(str(np.round(rv_weight,3))+'\t')

	f.write(target+'\t')
	f.write(template+'\n')

	f.close()

	return


def brvc(dateobs,exptime,observat,ra,dec,rv=0.0,print_out=False,epoch=2000,pmra=0,pmdec=0,px=0,query=False):
	'''
	observat options:
	- salt - (e.g. HRS)
	- eso22 - (e.g. FEROS)
	- vlt82 - (e.g. UVES)
	- mcdonald - (e.g. IGRINS)
	- lco_nres_lsc1 - (e.g. NRES at La Silla)
	- lco_nres_cpt1 - (e.g. NRES at SAAO)
	- tlv - (e.g. LCO NRES at Wise Observatory in Tel Aviv)
	- eso36 - (e.g. HARPS)
	- geminiS - (e.g. IGRINS South)
	- wiyn - (e.g. HYDRA)
	- dct - (e.g. IGRINS at DCT)
	- keck - (Keck Hi-Res)
	- smarts15 - (e.g. CHIRON)
	- hpf - (e.g. HPF)
	- keck - (e.g. HIRES)
	- irtf - (e.g. iSHELL)

	returns
	brv,bjd,bvcorr
	'''

	c = const.c.to('km/s').value

	from astroquery.vizier import Vizier  
	edr3 = Vizier(columns=["*","+_r","_RAJ2000","_DEJ2000","Epoch","Plx"],catalog=['I/350/gaiaedr3'])

	#from astroquery.gaia import Gaia
	
	if isinstance(observat,str):
		if isinstance(dateobs,str):
			n_sites = 1
			observat = [observat]
			dateobs = [dateobs]
			exptime = [exptime]
			rv = [rv]
		else:
			n_sites = dateobs.size
			observat = [observat]*dateobs.size
	else:
		n_sites = len(observat)

	brv = np.zeros(n_sites)
	bvcorr = np.zeros(n_sites)
	bjd = np.zeros(n_sites)

	#longitudes should be in degrees EAST!
	#Most are not unless they have a comment below them.
	for i in range(n_sites):
		if observat[i] == 'keck':
			alt = 4145
			lat = 19.82636 
			lon = -155.47501
			#https://latitude.to/articles-by-country/us/united-states/7854/w-m-keck-observatory#:~:text=GPS%20coordinates%20of%20W.%20M.%20Keck,Latitude%3A%2019.8264%20Longitude%3A%20%2D155.4750

		if observat[i] == 'gemini_south':
			alt = 2750
			lat = -30.24074167
			lon = -70.736683
			#http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon

		if observat[i] == 'salt':
			alt = 1798
			lat = -32.3794
			lon = 20.810694
			#http://www.sal.wisc.edu/~ebb/pfis/observer/intro.html
	
		if observat[i] == 'eso22':
			alt = 2335
			lat = -29.25428972
			lon = 289.26540472
			#FEROS Header
	
		if observat[i] == 'eso36':
			alt = 2400
			lat = -29.2584
			lon = 289.2565
			#HARPS - from a header
	
		if observat[i] == 'vlt82':
			alt = 2635
			lat = -24.6272
			lon = -70.4048
			#UVES HEAEDER
	
		if observat[i] == 'mcdonald':
			alt = 2075
			lat = 30.6716667
			lon = -104.0216667
			#https://idlastro.gsfc.nasa.gov/ftp/pro/astro/observatory.pro
	
		if observat[i] == 'wiyn':
			alt = 2120
			lat = 31.95222
			lon = -111.60000

		if observat[i] == 'dct':
			alt = 2360
			lat = 34.744444
			lon = -111.42194
			#https://en.wikipedia.org/wiki/Lowell_Discovery_Telescope

		if observat[i] == 'smarts15':
			alt = 2252.2
			lat = -30.169661
			lon = -70.806789
			#http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon

		if observat[i] == 'lsc':
			alt = 2201
			lat = -30.1673305556
			lon = -70.8046611111
			#From a header (CTIO - LCO)

		if observat[i] == 'cpt':
			alt = 1760.0
			lat = -32.34734167
			lon = 20.81003889
			#From a header (SAAO - LCO)

		if observat[i] == 'tlv':
			alt = 861.4
			lat = 30.595833
			lon = 34.763333
			#From a header (WISE, Isreal - LCO)

		if observat[i] == 'elp':
			alt = 2030.000
			lat = 30.6798330
			lon = -104.0151730
			#From a header (McDonald - LCO)

		if observat[i] == 'coj':
			alt = 1168.000
			lat = -31.2729330
			lon = 149.0706470
			#From a header (Siding Springs Observatory - LCO)

		if observat[i] == 'tres':
			alt = 2320.0
			lat = 31.68094444
			lon = -110.8775
			#From a TRES header 
			#1.5-meter Tillinghast telescope at the Smithsonian Astrophysical Observatory's Fred L. Whipple Observatory on Mt. Hopkins in Arizona

		if observat[i] == 'het':
			alt = 2026.0
			lat = 30.681389
			lon = 255.985278
			#https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json

		if observat[i] == 'lco-tfn':
			alt = 2390.0
			lat = 28.3003080
			lon = -16.5117030
			#from a header
			#LCO 0.4m Tenerife node

		if observat[i] == 'lco-ogg':
			alt = 3037.0
			lat = 20.7069583
			lon = -156.2575140
			#from a header
			#LCO 0.4m Haleakala node
	
		if observat[i] == 'irtf':
			alt = 4168.1
			lat = 19.826218
			lon = -155.471999
			#from a header

		if isinstance(ra,str):
			ra,dec = sex2dd(ra,dec)

		if query == True:
			coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
			
			result = edr3.query_region(coord,radius='0d0m3s')

			#Pgaia = Gaia.cone_search_async(coord, 3.0*u.arcsec)
			if len(result) == 0:
				print('No match found, using provided/default values.')
				epoch = 2000
				pmra = None
				pmdec = None
				px = None 
			else:
				dec = result['I/350/gaiaedr3']['DE_ICRS'][0]
				ra = result['I/350/gaiaedr3']['RA_ICRS'][0]
				epoch = result['I/350/gaiaedr3']['Epoch'][0]
				pmra = result['I/350/gaiaedr3']["pmRA"][0]
				pmdec = result['I/350/gaiaedr3']["pmDE"][0]
				px = result['I/350/gaiaedr3']["Plx"][0]
				if type(px) == np.ma.core.MaskedConstant:
					pmra = None
					pmdec = None
					px = None 

		if isinstance(dateobs[i],str):
			utc = Time(dateobs[i],format='isot',scale='utc')
		if isinstance(dateobs[i],float):
			utc = Time(dateobs[i],format='jd',scale='utc')

		utc_middle = utc + (exptime[i]/2.0)*u.second

		if observat in EarthLocation.get_site_names():
			bjd_info = utc_tdb.JDUTC_to_BJDTDB(JDUTC = utc_middle, obsname=observat, 
			                                   dec=dec, ra=ra, epoch=epoch, pmra=pmra, 
		                                       pmdec=pmdec, px=px)

			bvcorr_info = get_BC_vel(JDUTC = utc_middle, ra=ra, dec=dec, epoch=epoch, 
		                             pmra=pmra, pmdec=pmdec, px=px, obsname=observat)

		else:
			bjd_info = utc_tdb.JDUTC_to_BJDTDB(JDUTC = utc_middle, alt = alt, 
		                                       lat=lat, longi=lon, dec=dec, 
		                                       ra=ra, epoch=epoch, pmra=pmra, 
		                                       pmdec=pmdec, px=px)

			bvcorr_info = get_BC_vel(JDUTC = utc_middle, ra=ra, dec=dec, epoch=epoch, 
		                         	 pmra=pmra, pmdec=pmdec, px=px, lat=lat, longi=lon, 
		                         	 alt=alt)

		bjd[i] = bjd_info[0][0]
		bvcorr[i] = bvcorr_info[0][0]/1000.0

		if type(rv) == float:
			brv[i] = rv + bvcorr[i] + (rv * bvcorr[i] / (c))
		else:
			brv[i] = rv[i] + bvcorr[i] + (rv[i] * bvcorr[i] / (c))

	if print_out == True:
		print('BJD:  ',bjd)
		print('BRVC: ',bvcorr)
		print('BRV:  ',brv)

	return brv,bjd,bvcorr


def cont_norm(w,f,w_width=200.0,maxiter=15,lower=0.3,upper=2.0,nord=3,return_cont=False):
	'''
	Continuum normalizes a spectrum

	Uses the bspline_acr package which was adapted from IDL by 
	Aaron Rizzuto.

	Note: In the SAPHIRES architecture, this has to be done before 
	the spectrum is inverted, which happens automatically in the 
	saphires.io read in functions. That is why the option to 
	continuum normalize is available in those functions, and 
	should be done there. 

	Parameters
	----------
	w : array-like
		Wavelength array of the spectrum to be normalized.
		Assumed to be angstroms, but doesn't really matter.
	
	f : array-like
		Flux array of the spectrum to be normalized.
		Assumes linear flux units.
	
	w_width : number
		Width is the spline fitting window. This is useful for
		long, stitched spectra where is doesn't makse sense to 
		try and normalize the entire thing in one shot. 
		The defaults is 200 A, which seems to work reasonably well.
		Assumes angstroms but will naturally be on the same scale 
		as the wavelength array, w.

	maxiter : int
		Number of interations. The default is 15.

	lower : float 
		Lower limit in units of sigmas for including data in the 
		spline interpolation. The default is 0.3.

	upper : float
		Upper limit in units of sigma for including data in the 
		spline interpolation. The default is 2.0.

	nord : int
		Order of the spline. The defatul is 3.

	Returns
	-------
	f_norm : array-like
		Continuum normalized flux array

	'''
	norm_space = w_width #/(w[1]-w[0])

	#x = np.arange(f.size,dtype=float)
	spl = bspl.iterfit(w, f, maxiter = maxiter, lower = lower, 
	                   upper = upper, bkspace = norm_space, 
	      	           nord = nord )[0]
	cont = spl.value(w)[0]
	f_norm = f/cont

	if return_cont == False:
		return f_norm
	if return_cont == True:
		return f_norm,cont


def dd2sex(ra,dec,results=False):

    '''
    Convert ra and dec in decimal degrees format to sexigesimal format.

    Parameters:
    -----------
    ra : ndarray
        Array or single value of RA in decimal degrees.
        
    dec : ndarray
        Array or single value of Dec in decimal degrees.

    Results : bool
        If True, the decimal degree RA and Dec results are printed. 

    Returns:
    --------
    raho : ndarray
        Array of RA hours.

    ramo : ndarray
        Array of RA minutes.

    raso : ndarray
        Array of RA seconds.

    decdo : ndarray
        Array of Dec degree placeholders in sexigesimal format.

    decmo : ndarray
        Array of Dec minutes.

    decso : ndarray
        Array of Dec seconds.

    Output:
    -------
    Prints results to terminal if results == True.

    Version History:
    ----------------
    2015-05-15 - Start
    '''

    rah=(np.array([ra])/360.0*24.0)
    raho=np.array(rah,dtype=int)
    ramo=np.array(((rah-raho)*60.0),dtype=int)
    raso=((rah-raho)*60.0-ramo)*60.0
    dec=np.array([dec])
    dec_sign = np.sign(dec)
    decdo=np.array(dec,dtype=int)
    decmo=np.array(np.abs(dec-decdo)*60,dtype=int)
    decso=(np.abs(dec-decdo)*60-decmo)*60.0

    if results == True:
        for i in range(rah.size):
            print(str(raho[i])+':'+str(ramo[i])+':'+ \
                (str.format('{0:2.6f}',raso[i])).zfill(7)+', '+ \
                str(decdo[i])+':'+str(decmo[i])+':'+ \
                (str.format('{0:2.6f}',decso[i])).zfill(7))

    return raho,ramo,raso,dec_sign,decdo,decmo,decso


def d_gaussian_off(x,A1,x01,sig1,A2,x02,sig2,o):
    '''
    A double gaussian function with a constant vetical offset.

    Parameters
	----------
	x : array-like
		Array of x values over which the Gaussian profile will 
		be computed.

	A1 : float
		Amplitude of the first Gaussian profile. 

	x01 : float
		Center of the first Gaussian profile.

	sig1 : float
		Standard deviation (sigma) of the first Gaussian profile. 

	A2 : float
		Amplitude of the second Gaussian profile. 

	x02 : float
		Center of the second Gaussian profile.

	sig2 : float
		Standard deviation (sigma) of the second Gaussian profile. 

	o : float
		Vertical offset of the Gaussian mixture. 

    Returns
	-------
	profile : array-like
		The Gaussian mixture specified over the input x array.
		Array has the same length as x.
    '''
    return (A1*np.e**(-(x-x01)**2/(2.0*sig1**2))+
            A2*np.e**(-(x-x02)**2/(2.0*sig2**2))+o)


def EWM(x,xerr,wstd=False,variance_based=True):
    '''
    A function to return the error weighted mean of an array and the error on
    the error weighted mean. 

    Parameters
    ----------
    x : array like
       An array of values you want the error weighted mean of. 

    xerr : array like
       Array of associated one-sigma errors (variance based), or a relative weight
       (not variance based). Specify which with the variance_based option

    wstd : bool
        Option to return the error weighted standard deviation. If True, 
        three values are returned. The default is False.

    variance_based : bool
    	The weighted error calculation depends on whether the weights come from 
    	measurement errors (i.e., based on the variance of the measurement),
    	or some other non-variance, based weight. This keyword allows you to 
    	specify the nature of the xerr provided. 
    	Default is True

       
    Returns
    -------
    xmean : float
       The error weighted mean 
    
    xmeanerr : float
        The error on the error weighted mean. This number will only make
        sense if the input xerr is a 1-sigma uncertainty.

    xmeanstd : conditional output, float
        Error weighted standard deviation, only output if wstd=True

    Outputs
    -------
    None

    Version History
    ---------------
    2016-12-06 - Start
    '''
    weight = 1.0 / xerr**2
    xmean=np.sum(x/xerr**2)/np.sum(weight)

    if variance_based == True:
    	xmeanerr=1.0/np.sqrt(np.sum(weight))

    if variance_based == False:
    	weight_norm = weight / np.sum(weight)
    	sigma = np.std(x)
    	xmeanerr = sigma * np.sqrt(np.sum(weight_norm**2))

    xwstd = np.sqrt(np.sum(weight*(x - xmean)**2) / ((np.sum(weight)*(x.size-1)) / x.size))

    if wstd == True:
        return xmean,xmeanerr,xwstd
    else:
        return xmean,xmeanerr
        

def gaussian_off(x,A,x0,sig,o):
    '''
    A simple gaussian function with a constant vetical offset.
    This Gaussian is not normalized in the typical sense.

	Parameters
	----------
	x : array-like
		Array of x values over which the Gaussian profile will 
		be computed.

	A : float
		Amplitude of the Gaussian profile. 

	x0 : float
		Center of the Gaussian profile.

	sig : float
		Standard deviation (sigma) of the Gaussian profile. 

	o : float
		Vertical offset of the Gaussian profile. 

    Returns
	-------
	profile : array-like
		The Gaussian profile specified over the input x array.
		Array has the same length as x.
    '''

    return A*np.e**(-(x-x0)**2/(2.0*sig**2))+o


def lco2u(lco_fits,pkl_out,v2a=False):
	'''
	A function to read in the fits output from the standard 
	LCO/NRES pipeline and turn it into a pkl file that 
	matching the SAPHIRES architecture. 

	Parameters
	----------
	lco_fits : str
		The name of a single LCO fits file

	pkl_out : str
		The name of the output pickle file

	v2a:
		Option to convert the wavelength convention to air 
		from the provided vacuum values.

	Returns
	-------
	None

	'''
	hdu = pyfits.open(lco_fits)

	flux = hdu[3].data
	for i in range(flux.shape[0]):
		flux[i] = flux[i]+np.abs(np.min(flux[i]))

	w_vac = hdu[6].data*10

	if v2a == True:
		w_out = vac2air(w_vac)
	else:
		w_out = w_vac

	dict = {'wav':w_out, 'flux':flux}

	pkl.dump(dict,open(pkl_out,'wb'))

	print("LCO pickle written out to "+pkl_out)

	return


def make_rot_pro_ip(R,e=0.75):
	'''
	A function to make a specific rotationally broadened fitting 
	function with a specific limb-darkening parameter that is 
	convolved with the instrumental profile that corresponds to 
	a given spectral resolution.

	The output profile is uses the linear limb darkening law from
	Gray 2005

	Parameters
	----------
	R : float
		The resolution that corresponds to the spectrograph's 
		instrumental profile. 

	e : float
		Linear limb darkening parameter. Default is 0.75, 
		appropriate for a low-mass star.

	Returns
	-------
	rot_pro_ip : function
		A function that returns the line profile for a rotationally
		broadened star with the limb darkening parameter given by the make
		function and that have been convolved with the instrumental 
		profile specified by the spectral resolution by the make function 
		above.

		Parameters
		----------
		x : array-like
			X array values, should be provided in velocity in km/s, over
			which the smooth rotationally broadened profile will be 
			computed. 

		A : float
			Amplitude of the smoothed rotationally broadened profile. 
			Equal to the profile's integral.

		rv : float
			RV center of the profile. 

		rvw : float
			The vsini of the profile. 

		o : float
			The vertical offset of the profile. 

		Returns
		-------
		prof_conv : array-like
			The smoothed rotationally broadened profile specified by the
			paramteres above, over the input x array. Array has the same 
			length as x.

	'''
	c = const.c.to('km/s').value

	FWHM = (c)/R
	sig = FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))

	def rot_pro_ip(x,A,rv,rvw,o):
		'''
		A thorough descrition of this function is provieded in the main 
		function. 

		Rotational line broadening function. 
	
		To produce an actual line profile, you have to convolve this function
		with an acutal spectrum. 
	
		In this form it can be fit directly to a the Broadening Fucntion. 
	
		This is in velocity so if you're going to convolve this with a spectrum 
		make sure to take the appropriate cautions.
		'''

		c1 = (2*(1-e))/(np.pi*rvw*(1-e/3.0))
		c2 = e/(2*rvw*(1-e/3.0))
	
		prof=A*(c1*np.sqrt(1-((x-rv)/rvw)**2)+c2*(1-((x-rv)/rvw)**2))+o
	
		prof[np.isnan(prof)] = o
	
		v_spacing = x[1]-x[0]
	
		smooth_sigma = sig/v_spacing
	
		prof_conv=gaussian_filter(prof,sigma=smooth_sigma)
	
		return prof_conv

	return rot_pro_ip


def make_rot_pro_qip(R,a=0.3,b=0.4):
	'''
	A function to make a specific rotationally broadened fitting 
	function with a specific limb-darkening parameter that is 
	convolved with the instrumental profile that corresponds to 
	a given spectral resolution.

	The output profile is uses the linear limb darkening law from
	Gray 2005

	Parameters
	----------
	R : float
		The resolution that corresponds to the spectrograph's 
		instrumental profile. 

	e : float
		Linear limb darkening parameter. Default is 0.75, 
		appropriate for a low-mass star.

	Returns
	-------
	rot_pro_ip : function
		A function that returns the line profile for a rotationally
		broadened star with the limb darkening parameter given by the make
		function and that have been convolved with the instrumental 
		profile specified by the spectral resolution by the make function 
		above.

		Parameters
		----------
		x : array-like
			X array values, should be provided in velocity in km/s, over
			which the smooth rotationally broadened profile will be 
			computed. 

		A : float
			Amplitude of the smoothed rotationally broadened profile. 
			Equal to the profile's integral.

		rv : float
			RV center of the profile. 

		rvw : float
			The vsini of the profile. 

		o : float
			The vertical offset of the profile. 

		Returns
		-------
		prof_conv : array-like
			The smoothed rotationally broadened profile specified by the
			paramteres above, over the input x array. Array has the same 
			length as x.

	'''
	c = const.c.to('km/s').value

	FWHM = (c)/R
	sig = FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))

	def rot_pro_qip(x,A,rv,rvw,o):
		'''
		A thorough descrition of this function is provieded in the main 
		function. 

		Rotational line broadening function. 
	
		To produce an actual line profile, you have to convolve this function
		with an acutal spectrum. 
	
		In this form it can be fit directly to a the Broadening Fucntion. 
	
		This is in velocity so if you're going to convolve this with a spectrum 
		make sure to take the appropriate cautions.
		'''

		prof = A*((2.0*(1-a-b)*np.sqrt(1-((x-rv)/rvw)**2) + 
		           np.pi*((a/2.0) + 2.0*b)*(1-((x-rv)/rvw)**2) - 
		           (4.0/3.0)*b*(1-((x-rv)/rvw)**2)**(3.0/2.0)) / 
		         (np.pi*(1-(a/3.0)-(b/6.0)))) + o
		
		prof[np.isnan(prof)] = o
	
		v_spacing = x[1]-x[0]
	
		smooth_sigma = sig/v_spacing
	
		prof_conv=gaussian_filter(prof,sigma=smooth_sigma)
	
		return prof_conv

	return rot_pro_qip


def order_stitch(t_f_names,spectra,n_comb,print_orders=True):
	'''
	A function to stitch together certain parts a specified 
	number of orders throughout a dictionary.
	e.g. an 8 order spectrum, with a specified number of orders
	to combine set to 2, will stich together 1-2, 3-4, 5-6, and 7-8,
	and result in a dictionary with 4 stitched orders. 

	If the number of combined orders does not divide evenly, the 
	remaining orders will be appended to the last stitched order.
	'''
	n_orders = t_f_names[t_f_names!='Combined'].size

	n_orders_out = int(n_orders/float(n_comb))

	spectra_out = {}
	t_f_names_out = np.zeros(n_orders_out,dtype=nplts+'1000')

	for i in range(n_orders_out):
		w_all = np.empty(0)
		flux_all = np.empty(0)

		for j in range(n_comb):
			w_all = np.append(w_all,spectra[t_f_names[i*n_comb+j]]['nwave'])
			flux_all = np.append(flux_all,spectra[t_f_names[i*n_comb+j]]['nflux'])
			
			if j == 0:
				w_range_all = spectra[t_f_names[i*n_comb+j]]['w_region']+','
			if ((j > 0) & (j<n_comb-1)):
				w_range_all = w_range_all+spectra[t_f_names[i*n_comb+j]]['w_region']+','
			if j == n_comb-1:
				w_range_all = w_range_all+spectra[t_f_names[i*n_comb+j]]['w_region']
		
		if i == n_orders_out-1:
			leftover = n_orders - (i*n_comb+n_comb)
			for j in range(leftover):
				w_all = np.append(w_all,spectra[t_f_names[i*n_comb+n_comb+j]]['nwave'])
				flux_all = np.append(flux_all,spectra[t_f_names[i*n_comb+n_comb+j]]['nflux'])
				
				if j == 0:
					w_range_all = w_range_all+','+spectra[t_f_names[i*n_comb+n_comb+j]]['w_region']+','
				if ((j > 0) & (j < leftover-1)):
					w_range_all = w_range_all+spectra[t_f_names[i*n_comb+n_comb+j]]['w_region']+','
				if j == leftover-1:
					w_range_all = w_range_all+spectra[t_f_names[i*n_comb+n_comb+j]]['w_region']

		if w_range_all[-1] == ',':
			w_range_all = w_range_all[:-1]

		flux_all = flux_all[np.argsort(w_all)]
		w_all = w_all[np.argsort(w_all)]

		w_min=int(np.min(w_all))
		w_max=int(np.max(w_all))

		t_dw = np.median(w_all - np.roll(w_all,1))

		t_f_names_out[i] = ('R'+str(i)+'['+str(i)+']['+str(w_min)+'-'+str(w_max)+']')

		if print_orders == True:
			print(t_f_names_out[i],w_range_all)
		spectra_out[t_f_names_out[i]] = {'nflux': flux_all,
										 'nwave': w_all,
										 'ndw': np.median(np.abs(w_all - np.roll(w_all,1))),
										 'wav_cent': np.mean(w_all),
										 'w_region': w_range_all,
										 'rv_shift': spectra[t_f_names[0]]['rv_shift'],
										 'order_flag': 1}
		
	return t_f_names_out,spectra_out


def prepare(t_f_names,t_spectra,temp_spec,oversample=1,quiet=False,trap_apod=0,cr_trim=-0.2,cr_trim_temp=-0.2,trim_style='clip',vel_spacing='uniform',set_spacing=False):
	'''
	A function to prepare a target spectral dictionary with a template 
	spectral dictionary for use with SAPHIRES analysis tools. The preparation
	ammounts to resampling the wavelength array to logarithmic spacing, which 
	corresponds to linear velocity spacing. Linear velocity spacing is required 
	for use TODCOR or compute a broadening function
	
	oversample - 
	A key parameter of this function is "oversample". It sets the logarithmic 
	spacing of the wavelength resampling (and the corresponding velocity 
	'resolution' of the broadening function). Generally, a higher oversampling
	will produce better results, but it very quickly becomes expensive for two
	reasons. One, your arrays become longer, and two, you have to increase the m
	value (a proxy for the velocity regime probed) to cover the same velocity range. 

	A low oversample value can be problematic if the intrinsitic width of your
	tempate and target lines are the same. In this case, the broadening function 
	should be a delta function. The height/width of this function will depened on 
	the location of velocity grid points in the BF in an undersampled case. This 
	makes measurements of the RV and especially flux ratios problematic. 

	If you're unsure what value to use, look at the BF with low smoothing
	(i.e. high R). If the curves are jagged and look undersampled, increase the 
	oversample parameter.

	Parameters
	----------
	t_f_names: array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum. Output of one of the 
		saphires.io read-in functions.

	temp_spec : python dictionary
		SAPHIRES dictionary for the template spectrum. Output of one of the 
		saphires.io read-in functions.		

	oversample : float
		Factor by which the velocity resolution is oversampled. This parameter
		has an extended discussion above. The default value is 1.

    quiet : bool
    	Specifies whether messaged are printed to the teminal. Specifically, if 
    	the science and template spectrum do not overlap, this function will
    	print and error. The default value is False.

    trap_apod : float
    	Option to apodize (i.e. taper) the resampled flux array to zero near 
    	the edges. A value of 0.1 will taper 10% of the array length on each 
    	end of the array. Some previous studies that use broaden fuctions in 
    	the literarure use this, claiming it reduced noise in the sidebands. I
    	havent found this to be the case, but the functionallity exisits 
    	nonetheless. The detault value is 0, i.e. no apodization.

    cr_trim	: float
		This parameter sets the value below which emission features are removed
		from the target (observed/science) spectrum. 
		Emission is this case is negative becuase the spectra are inverted. The
		value must be negative. Points below this value are linearly interpolated
		over. The defulat value is -0.2. If you don't want to clip anything, set 
		this paramter to -np.inf.

	cr_trim_temp : float
		This parameter sets the value below which emission features are removed
		from the template (narrow-lined) spectrum. 
		Emission is this case is negative becuase the spectra are inverted. The
		value must be negative. Points below this value are linearly interpolated
		over. The defulat value is -0.2. If you don't want to clip anything, set 
		this paramter to -np.inf.

    trim_style : str, options: 'clip', 'lin', 'spl'
		If a wavelength region file is input in the 'spectra_list' parameter, 
		this parameter describes how gaps are dealt with. 
		- If 'clip', unused regions will be left as gaps.
		- If 'lin', unused regions will be linearly interpolated over.
		- If 'spl', unused regions will be interpolated over with a cubic 
		  spline. You probably don't want to use this one.

	vel_spacing : str; 'orders' or 'uniform', or float
		Parameter that determines how the velocity width of the resampled array 
		is set. 
		If 'orders', the velocity width will be set by the smallest velocity
		separation between the native input science and template wavelength arrays on
		an order by order basis.
		If 'uniform', every order will have the same velocity spacing. This is useful
		if you want to combine BFs for instance. The end result will generally be 
		slightly oversampled, but other than taking a bit longer, to process, it should
		not have any adverse effects. 
		If this parameter is a float, the velocity spacing will be set to that value,
		assuming it is in km/s. 
		You can get wierd results if you put in a value that doesn't make sense, so 
		I recommend the orders or uniform setting. This option is available for more 
		advanced use cases that may only relevant if you are using TODCOR. See 
		documentation there for a relevant example. 
		The oversample parameter is ignored when this parameter is set to a float.


	Returns
	-------
	spectra : dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will have 5 new keywords as a result of this function.

		['vflux'] 		- resampled flux array (inverted)					
		['vwave'] 		- resampled wavelength array							
		['vflux_temp']	- resampled template flux array (inverted)			
		['vel'] 		- velocity array to be used with the BF or CCF		
		['temp_name'] 	- template name		
		['vel_spacing'] - the velocity spacing that corresponds to the 
						  resampled wavelength array

		It also updates the values for the following keyword under the right 
		conditions:

		['order_flag'] 	- order flag will be updated to 0 if the order has no 
						  overlap with the template. This tells other functions
						  to ignore this order. 


	'''
	#########################################
	#This part "prepares" the spectra

	c = const.c.to('km/s').value

	spectra = copy.deepcopy(t_spectra)

	max_w_orders = np.zeros(t_f_names.size)
	min_w_orders = np.zeros(t_f_names.size)
	min_dw_orders = np.zeros(t_f_names.size)

	for i in range(t_f_names.size):
		w_tar = spectra[t_f_names[i]]['nwave']
		flux_tar = spectra[t_f_names[i]]['nflux']
		w_range = spectra[t_f_names[i]]['w_region']
	
		w_temp = temp_spec['nwave']
		flux_temp = temp_spec['nflux']
		temp_trim = temp_spec['w_region']
	
		w_tar,flux_tar = spec_trim(w_tar,flux_tar,w_range,temp_trim,trim_style=trim_style)
		
		w_temp,flux_temp = spec_trim(w_temp,flux_temp,w_range,temp_trim,trim_style=trim_style)
		
		if w_tar.size == 0:
			min_w_orders[i] = np.nan
			max_w_orders[i] = np.nan
			min_dw_orders[i] = np.nan
		else:
			min_w_orders[i] = np.max([np.min(w_tar),np.min(w_temp)])
			max_w_orders[i] = np.min([np.max(w_tar),np.max(w_temp)])
			min_dw_orders[i]=np.min([temp_spec['ndw'],spectra[t_f_names[i]]['ndw']])
	
	min_dw = np.nanmin(min_dw_orders)
	min_w = np.nanmin(min_w_orders)
	max_w = np.nanmax(max_w_orders)

	if vel_spacing == 'uniform':
		r = np.min(min_dw/max_w/oversample)
		#velocity spacing in km/s
		stepV=r*c

	if ((type(vel_spacing) == float) or isinstance(vel_spacing,float)):
		stepV = vel_spacing
		r = stepV / (c)
		min_dw = r * max_w

	for i in range(t_f_names.size):
		spectra[t_f_names[i]]['temp_name'] = temp_spec['temp_name']
		
		w_range = spectra[t_f_names[i]]['w_region']

		w_tar = spectra[t_f_names[i]]['nwave']
		flux_tar = spectra[t_f_names[i]]['nflux']
		
		temp_trim = temp_spec['w_region']
		w_temp = temp_spec['nwave']
		flux_temp = temp_spec['nflux']
		
		#This gets rid of large emission lines and CRs by interpolating over them.
		flux_tar,w_tar = cr_trim_spec(flux_tar,w_tar,cr_trim)
		#if np.min(flux_tar) < cr_trim:
		#	f_tar = interpolate.interp1d(w_tar[flux_tar > cr_trim],flux_tar[flux_tar > cr_trim])
		#	w_tar = w_tar[(w_tar >= np.min(w_tar[flux_tar > cr_trim]))&
		#				  (w_tar <= np.max(w_tar[flux_tar > cr_trim]))]
		#	flux_tar = f_tar(w_tar)

		flux_temp,w_temp = cr_trim_spec(flux_temp,w_temp,cr_trim_temp)
		#if np.min(flux_temp) < cr_trim_temp:
		#	f_temp = interpolate.interp1d(w_temp[flux_temp > cr_trim_temp],flux_temp[flux_temp > cr_trim_temp])
		#	w_temp = w_temp[(w_temp >= np.min(w_temp[flux_temp > cr_trim_temp]))&
		#					(w_temp <= np.max(w_temp[flux_temp > cr_trim_temp]))]
		#	flux_temp = f_temp(w_temp)

		w_tar,flux_tar = spec_trim(w_tar,flux_tar,w_range,temp_trim,trim_style=trim_style)

		if w_tar.size == 0:
			if quiet==False:
				print(t_f_names[i],w_range)
				print("No overlap between target and template.")
				print(' ')
			spectra[t_f_names[i]]['vwave'] = 0.0
			spectra[t_f_names[i]]['order_flag'] = 0
			spectra[t_f_names[i]]['vel_spacing'] = stepV
			continue

		f_tar = interpolate.interp1d(w_tar,flux_tar)
		f_temp = interpolate.interp1d(w_temp,flux_temp)

		if vel_spacing == 'orders':
			#Using the wavelength spacing of the most densely sampled spectrum
			min_dw=np.min([temp_spec['ndw'],spectra[t_f_names[i]]['ndw']])
	
			#inverse of the spectral resolution
			r = min_dw/max_w/oversample 
	
			#velocity spacing in km/s
			stepV = r * c

		if set_spacing == False:
			min_w = np.max([np.min(w_tar),np.min(w_temp)])

			max_w = np.min([np.max(w_tar),np.max(w_temp)])

			#the largest array length between target and spectrum
			#conditional below makes sure it is even
			max_size = int(np.log(max_w/(min_w+1))/np.log(1+r))
			if (max_size/2.0 % 1) != 0:
				max_size=max_size-1
		else:
			min_w,max_size = set_spacing
			if (max_size/2.0 % 1) != 0:
				max_size=max_size-1


		#log wavelength spacing, linear velocity spacing
		w1t=(min_w+1)*(1+r)**np.arange(max_size)
		
		w1t_temp = copy.deepcopy(w1t)

		t_rflux = f_tar(w1t)
		temp_rflux = f_temp(w1t)

		w1t,t_rflux = spec_trim(w1t,t_rflux,w_range,temp_trim,trim_style=trim_style)

		w1t_temp,temp_rflux = spec_trim(w1t_temp,temp_rflux,w_range,temp_trim,trim_style=trim_style)

		if (w1t.size/2.0 % 1) != 0:
			w1t=w1t[0:-1]
			t_rflux = t_rflux[0:-1]
			temp_rflux = temp_rflux[0:-1]

		if trap_apod > 0:
			trap_apod_fun = np.ones(w1t.size)
			slope = 1.0/int(w1t.size*trap_apod)
			y_int = slope*w1t.size
			trap_apod_fun[:int(w1t.size*trap_apod)] = slope*np.arange(int(w1t.size*trap_apod),dtype=float)
			trap_apod_fun[-int(w1t.size*trap_apod)-1:] = -slope*(np.arange(int(w1t.size*trap_apod+1),dtype=float)+(w1t.size*(1-trap_apod))) + y_int

			temp_rflux = temp_rflux * trap_apod_fun
			t_rflux = t_rflux * trap_apod_fun

		spectra[t_f_names[i]]['vflux'] = t_rflux
		spectra[t_f_names[i]]['vwave'] = w1t
		spectra[t_f_names[i]]['vflux_temp'] = temp_rflux
		spectra[t_f_names[i]]['vel_spacing'] = stepV
		spectra[t_f_names[i]]['w_region_temp'] = temp_spec['w_region']

	return spectra


def cr_trim_spec(f,w,val):
	if type(val) == float:
		if np.min(f) < val:
			f_f = interpolate.interp1d(w[f > val],f[f > val])
			w = w[(w >= np.min(w[f > val]))&(w <= np.max(w[f > val]))]
			f = f_f(w)

	if type(val) == str:
		if val == 'adapt':
			val_i = -1.5
			step = 0.05
			thresh = 0.005
			frac = np.sum((f < val_i))/float(f.size)

			while frac < thresh:
				val_i = val_i + step
				frac = np.sum((f < val_i))/float(f.size)
			
			f_f = interpolate.interp1d(w[f > val_i],f[f > val_i])
			w = w[(w >= np.min(w[f > val_i]))&(w <= np.max(w[f > val_i]))]
			f = f_f(w)

		else:
			print(val+" is not an accepted keyword for cf_trim. Try 'adapt'.")

	return f,w


def RChiS(x,y,yerr,func,params):
	'''
	A function to compute the Reduced Chi Square between some data and 
	a model. 

	Parameters
	----------
	x : array-like
		Array of x values for data.

	y : array-like
		Array of y values for data.

	yerr : array-like
		Array of 1-sigma uncertainties for y vaules. 

	func : function
		Function being compared to the data.

	params : array-like
		List of parameter values for the function above.

	Returns
	-------
	rchis : float
		The reduced chi square between the data and model.

	'''
	rchis=np.sum((y-func(x,*params))**2 / yerr**2 )/(x.size-params.size)

	return rchis


def line(x,a,b):

	return a*x + b


def li_analysis(p_file,teff,logg,rv,vsini,order=18,w_conv='vac',R=40000.0,vel_buffer=1.8,sigma_cut = 2,up_cut=1.5,path='/Users/bt9644/templates/PHOENIX/'):
	'''
	Interactive function to measure li EWs.

	This function is appropriate for stars that have continuum. Mid M dwarfs
	and later would be better suited to a empirical template subtraction 
	method.

	This function plots your spectrum and overlays a model of a given teff, logg,
	rv, vsini, and instrumental broadening, to give some context for where the line
	is and where continuum regions are. This is useful for rotationally broadened 
	stars where weak lines bias your continuum level to lower values. 

	Running the script pops up an interactive plotting window with the specrum
	and the broadened tempalte above. Red points in the observed spectrum are the 
	initial values that are used to define the continuum (from the sigma_cut 
	parameter), which is shown as the horizontal blue line. 

	Here you have the option to use the default values (y), create your own (n), or
	quit (q).

	If 'n', use the b key to mark continuum regions from left to right and the m key
	mark the region you want to preform the numerical intergration of the EW, again,
	left to right. Hit return to continue

	The code then fits the continuum regions with a line via mcmc. The slope and 
	y-intercept posteriors are randomly drawn from to normalize the spectrum 500 
	times. In each of the 500 iterations the EW is numertically integrated over 
	the specified window, where the bounds vary randomly over a resolution element. 
	Gaussian noise is randomly added to the integration region at the level of the 
	variance in the continuum regions.

	The goal here is vary the things that go into the computation. 

	The 500 bootstrap values are used to define the 50th percentile and 68% confidence
	interval. 

	A plot is output that shows the spectral regions used and the EW bootstrap 
	posterior.

	Parameters
	----------
	p_file : str or tuple
		Name of a standard saphires spectral pickle, which is a 
		dictionary with of wavelength and flux arrays with 'wav'
		and 'flux' keywords, respectively. The arrays associated with 
		each should have the shape of [values in a given order,order]
		OR
		A tuple with (wavelength,flux) arrays. They should be for the 
		that contains the Li 6708 line.

	teff : str
		The teff of the star in question. Used in pulling a PHOENIX templatr
		from the saphires spectral template library

	logg : str
		The logg of the star in question. Used in pulling a PHOENIX templatr
		from the saphires spectral template library

	rv : float
		The observed rv of the star (not corrected for heliocentric motion). 
		Used to shift the stellar spectrum to zero velocity.

	vsini : float
		The vsini of the star. Used to rotationally broaden the template

	order : int
		The order that has Li 6708. Only used if the p_file is a saphires
		pickled dictionary. If you provide p_file with a (w,f) tuple, this 
		parameter is ignored.
		Default is 18, which is the Li order for LCO/NRES

	w_conv : str
		String defining the wavelength convention of the observed spectrum
		Options are 'vac' or 'air'
		Used to define the location of the line and pull the appropriate 
		synthetic template from the saphires spectral template library
		Default is 'vac', for LCO/NRES

	R : float
		Resolution of the spectrograph. Used to broaden the template
		Default is 40000.0, for LCO/NRES

	vel_buffer : float
		Defines a extra buffer outside of the vsini of the line, which is 
		used to set the width of the numerical integration.
		Default is 1.8.

	sigma_cut : int
		Parameter for automated continuum region selection. Removes areas
		that are a number sigma outliers given to this parameter.
		Default is 2.

	up_cut : float
		Removed values from the observed/continuum normalilzed spectrum 
		that have flux points above the given value. For removing suprious 
		values. 
		Default is 1.5

	path : str
		Local path to the saphires spectral template library.


	Scripring usage
	---------------
	in terminal setup:
	>>> cat *mcmc.dat > li_head_in.dat
	>>> p_file_long,rv,vsini = np.loadtxt('li_head_in.dat',unpack=True,usecols=(0,7,8),dtype='U100,f,f',delimiter=',')
	>>> for i in range(p_file_long.size):
			p_file = p_file_long[i][:-11]+'.pkl'
			teff = p_file_long[i][-10:-6]
			logg = p_file_long[i][-5:-2]
			saph.utils.li_analysis(p_file,teff,logg,rv[i],vsini[i])

	Notes
	-----
	Default continuum regions do an interative 3-sigma clip to remove detected line. Seems to
	do a good job, but in principle it will underestimate the continuum level due to undetected
	shallow lines, which will underestimate the li EW. So far, this has been less than the errors. 

	(For SB2s, this is the best thing to do because choosing continuum regions from a zero-velocity 
	 template doesn't make sense when there are two sets of lines.)

	You have the option to select the continuum regions by hand, which might give you better results 
	in certain scenarios
	- trying to measure faint features
	- when the spectrum is noisy and the vsini is small, narrow lines you can tell are there but are
	  not picked up by the default version.

	'''

	def press_key(event):
		if event.key == 'b':
			l_range.append(np.round(event.xdata,2))

			if (len(l_range)/2.0 % 1) != 0:
				ax.axvline(event.xdata,ls='-',color='k')
			else:
				ax.axvline(event.xdata,ls='--',color='k')
			plt.draw()

			return l_range

		if event.key == 'm':
			i_range.append(np.round(event.xdata,2))
			ax.axvline(event.xdata,ls=':',lw=3,color='r')
			plt.draw()

			return

	c = const.c.to('km/s').value

	if w_conv == 'vac':
		li = air2vac(6707.81)
		li_hi = air2vac(6707.91)
		li_lo = air2vac(6707.76)
	else:
		li = 6707.81
		li_hi = 6707.91
		li_lo = 6707.76

	width_kms = np.max([vsini*vel_buffer,30])

	wi_start = (-width_kms)*li_lo/(c) + li_lo
	wi_end = (width_kms)*li_hi/(c) + li_hi

	#-------------------------------------------------------------------
	#getting the model read in, continuum normalized, and broadeded
	#-------------------------------------------------------------------
	syn = pkl.load(open(path+'logg'+logg+'/lte0'+teff+'-'+logg+'0-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_2800-11000_'+w_conv+'.p','rb'),encoding='latin')
	
	w_syn_ha = syn['wav']
	f_syn_ha = syn['flux']
	
	w_syn = syn['wav']
	f_syn = syn['flux']

	f_syn = f_syn[(w_syn > 6400) & (w_syn < 6900)]
	w_syn = w_syn[(w_syn > 6400) & (w_syn < 6900)]


	f_syncn = cont_norm(w_syn,f_syn,w_width=250.0,maxiter=15,lower=0.45,upper=2.0,nord=3)
	
	f_syncn = f_syncn[(w_syn > 6685) & (w_syn < 6735)]
	w_syn = w_syn[(w_syn > 6685) & (w_syn < 6735)]
	
	f_f_syn = interpolate.interp1d(w_syn,f_syncn)
	
	r = (w_syn[1]-w_syn[0])/np.max(w_syn)
	stepV = r * c
	max_size = int(np.log(np.max(w_syn)/(np.min(w_syn)+1))/np.log(1+r))
	if (max_size/2.0 % 1) != 0:
		max_size=max_size-1
	
	w_syn_ls = (np.min(w_syn)+1)*(1+r)**np.arange(max_size)
	
	f_syncn_ls = f_f_syn(w_syn_ls)
	
	vel = np.linspace(-100,100,int(200/stepV))
	
	rot_pro_ip_no = make_rot_pro_ip(R,e=0.6)
	
	rot_model = rot_pro_ip_no(vel,1.0,0,vsini,0)
	
	f_syn_broad = 1.0-np.convolve(1.0-f_syncn_ls,rot_model,mode='same')

	#------------------------------------------------------------------
	# read in the spectra pkl
	#------------------------------------------------------------------
	if isinstance(p_file,str):
		hdu = pkl.load(open(p_file,'rb'))
		f = hdu['flux'][order,:]
		w_i = hdu['wav'][order,:]

	if isinstance(p_file,tuple):
		f = p_file[0]
		w_i = p_file[1]
	
	w_i = w_i/(1-(-rv/(c)))
	
	f = f[~np.isnan(w_i)]
	w_i = w_i[~np.isnan(w_i)]
	
	w_i = w_i[(~np.isnan(f)) & (f != 0)]
	f = f[(~np.isnan(f)) & (f != 0)]

	f_norm = cont_norm(w_i,f,w_width=50.0,maxiter=20,lower=0.5,upper=0.5,nord=5)

	w_i = w_i[f_norm < up_cut]
	f_norm = f_norm[f_norm < up_cut]


	#------------------------------------------------------------------
	#default continuum regions:
	#------------------------------------------------------------------

	outliers = True

	cont_regions = (((w_i > 6685) & (w_i < wi_start)) | ((w_i > wi_end) & (w_i < 6730)))

	while outliers == True:
		line_fit,line_fit_cov = curve_fit(line,w_i[cont_regions],f_norm[cont_regions])

		resid = (f_norm/line(w_i,*line_fit)-1.0) / np.std(f_norm/line(w_i,*line_fit))

		if np.min(resid[cont_regions]) < -sigma_cut:
			outliers = True
			cont_regions = cont_regions * (resid > -sigma_cut)
			
			#print(np.sum(cont_regions))

		else:
			outliers = False

	print(np.sum(cont_regions))

	#------------------------------------------------------------------
	# plot the li region and define the continuum regions
	#------------------------------------------------------------------
	plt.ion()

	fig,ax = plt.subplots(1,figsize=(14.25,7.5))

	ax.plot(w_syn_ls,f_syncn_ls+1,alpha=0.75,color='grey',lw=0.5)
	#ax.plot(w_syn_ls[cont_regions_syn],f_syn_broad[cont_regions_syn]+1,'o',alpha=0.75,color='C0')
	ax.plot(w_syn_ls,f_syn_broad+1,alpha=0.75,color='C1',lw=2)

	ax.plot(w_i,f_norm,color='C2')
	ax.plot(w_i[cont_regions],f_norm[cont_regions],'o',color='C3',ms=3)

	ax.axhline(1-np.std(f_norm[cont_regions]/line(w_i[cont_regions],*line_fit)))

	ax.axvline(wi_start,ls=':',color='red')
	ax.axvline(wi_end,ls=':',color='red')
	ax.axvline(li,ls='--',color='red')

	ax.set_xlim(6685,6730)
	ax.set_ylim(0.0,2.5)

	ax.set_ylabel('Normalized Flux')
	ax.set_xlabel('Wavelength ($\AA$)')

	plt.tight_layout()

	print('')
	print('------')
	print(p_file)
	use_default = p_input('Use the default windows (continuum in red dots, integration within lines), or quit? (y,n,q): ')
	print('')

	while use_default not in ['y','n','q']:
		use_default = p_input('try again (y,n,q): ')

	if use_default == 'y':

		plt.cla()
		
		plt.close('all')

	if use_default == 'n':
		print('Use b to mark continuum regions, m to mark integration range.')
		print('Once done, press return to contiune.')

		l_range = []
		i_range = []
		
		cid = fig.canvas.mpl_connect('key_press_event',press_key)
		
		wait = p_input('')
	
		fig.canvas.mpl_disconnect(cid)
		
		plt.cla()
		
		plt.close('all')
    	    
		#cont_regions_syn = np.zeros(w_syn_ls.size,dtype=bool)
		cont_regions = np.zeros(w_i.size,dtype=bool)
	
		if len(l_range) > 0:
			for i in range(len(l_range)//2):
				#cont_regions_syn[(w_syn_ls > l_range[i*2]) & (w_syn_ls < l_range[i*2+1])] = True
				cont_regions[(w_i > l_range[i*2]) & (w_i < l_range[i*2+1])] = True

		if len(i_range) > 0:
			if len(i_range) == 2:
				wi_start = i_range[0]
				wi_end = i_range[1]
			else:
				print("Require only one start and end for the integration range. Try again...")
				return 

	if use_default == 'q':
		print('On to the next one!')
		print('---------')
		print('')

		plt.cla()
		plt.close('all')
		
		return


	#------------------------------------------------------------------
	# fit the continuum regions
	#------------------------------------------------------------------

	line_fit,line_fit_cov = curve_fit(line,w_i[cont_regions],f_norm[cont_regions])

	f_mcn = f_norm/line(w_i,*line_fit)

	ferr_m = line(w_i,*line_fit) * np.std(f_mcn[cont_regions])

	f_m_std = np.std(f_mcn[cont_regions])

	def log_likelihood(params, w, obs, obs_err):
		a,b,log_f = params
		model = line(w,a,b)
		sigma2 = obs_err**2 + model**2*np.exp(2*log_f)
		
		return -0.5*np.sum((obs-model)**2/sigma2 + np.log(sigma2))
		
	def log_prior(params):
		a,b,log_f = params
	
		if -10 < a < 10 and -100 < b < 100 and -10.0 < log_f < 1.0:
		    return 0.0
		else:
		    return -np.inf
	
	def log_probability(params, w, obs, obs_err):
		lp = log_prior(params)
		if not np.isfinite(lp):
		    return -np.inf
		return lp + log_likelihood(params, w, obs, obs_err)

	max_steps = 10000


	mcmc_start = np.array([line_fit[0],line_fit[1],-9])
		
	pos = mcmc_start + np.array([1e-4,1e-4,1e-4])*np.random.randn(32, 3)
	nwalkers, ndim = pos.shape
		
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
	                                args=(w_i[cont_regions],f_norm[cont_regions],ferr_m[cont_regions]),
	                                threads=7)
	
	index = 0
	autocorr = np.empty(max_steps)
	N = np.empty(max_steps)
	old_tau = np.inf
	converged = False
	# Now we'll sample for up to max_n steps
	for sample in sampler.sample(pos, iterations=max_steps, progress=False):
		# Only check convergence every 100 steps
		if converged == False:
			if sampler.iteration % 1000 == 0:
				print('Step count:',sampler.iteration)
			if sampler.iteration % 100:
			    continue
		
			#Measure convergence 
			tau = sampler.get_autocorr_time(tol=0)
			autocorr[index] = np.mean(tau)
			N[index] = sampler.iteration
		
			#Check convergence
			converged = np.all(tau * 50 < sampler.iteration)
			converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
			if converged:
				#converge_step = sampler.chain.shape[1]
				tau_converge = sampler.get_autocorr_time()
				burnin = int(5 * np.max(tau_converge))
				break
			old_tau = tau
			index += 1
	if converged == False:
		print('Unable to measure convergence')
		burnin = int(max_steps/4.0)
	N = N[:index-1]
	autocorr = autocorr[:index-1]
	samples = sampler.chain
	corner_samples = sampler.get_chain(discard=burnin, flat=True)
		

	a_samples = samples[:,burnin:,0].flatten()
	b_samples = samples[:,burnin:,1].flatten()
	
	labels=['a','b','jitter']
	
	a = np.percentile(a_samples,50,axis=0)
	b = np.percentile(b_samples,50,axis=0)
	
	f_mcn_mcmc = f_norm/line(w_i,a,b)
	
	n_iters = 10
	
	int_ind = ((w_i > wi_start) & (w_i < wi_end))

	#------------------------------------------------------
	# now bootstrap over a bunch of stuff
	#------------------------------------------------------
	n_steps = 500

	ew_dist = np.zeros(n_steps*n_iters)
	ew_gdist = np.zeros(n_steps*n_iters)
	
	vres = c/R
	vsig = vres/(2.0*np.sqrt(2.0*np.log(2.0)))

	k=0
	for i in range(n_steps):
		for j in range(n_iters):
			#wi_start = (-width_kms)*li_lo/(c) + li_lo
			#wi_start = (-(width_kms+vres*2.0) + np.random.normal(0,vres))*li_lo/(c) + li_lo
			wi_start_i = wi_start - (li*((2*vsig+np.random.normal(0,vsig))/c))
			wi_end_i = wi_end + (li*((2*vsig+np.random.normal(0,vsig))/c))
			#wi_end = (width_kms+vres*2.0 + np.random.normal(0,vres))*li_hi/(c) + li_hi
	
			int_ind = ((w_i > wi_start_i) & (w_i < wi_end_i))
	
			f_m_i = f_norm + np.random.normal(0,f_m_std,f_norm.size)
	
			ew_dist[k] = trap_int(w_i[int_ind],1.0-(f_m_i/line(w_i,a_samples[i],b_samples[i]))[int_ind])[0]

			k = k + 1
	
	ew = np.percentile(ew_dist,[16,50,84],axis=0)
	
	print('')
	print('Li EW:',ew[1],ew[2]-ew[1],ew[0]-ew[1])
	print('--------')
	print('')


	#-------------------------------------------------------------------
	#----------- PLOTS -------------------------------------------------
	#-------------------------------------------------------------------
	
	pp = PdfPages(p_file.split('.')[0]+'_li.pdf')
	
	fig,ax = plt.subplots(1)

	ax.plot(w_syn,f_syncn+1,alpha=0.75,color='grey')
	#ax.plot(w_syn_ls[cont_regions_syn],f_syn_broad[cont_regions_syn]+1,'o',alpha=0.75,color='C0')
	ax.plot(w_syn_ls,f_syn_broad+1,color='C1')
	
	ax.plot(w_i,f_norm,color='C2')
	ax.plot(w_i[cont_regions],f_norm[cont_regions],'o',color='C3',ms=3)
	
	rn_ind = np.random.randint(0,n_steps,50)
	for i in range(rn_ind.size):
		ax.plot(w_i,line(w_i,a_samples[rn_ind[i]],b_samples[rn_ind[i]]),color='lightgrey',alpha=0.2)
	ax.plot(w_i,line(w_i,*line_fit),color='grey')

	ax.axvline(wi_start,ls=':',color='k')
	ax.axvline(wi_end,ls=':',color='k')
	ax.axvline(li,ls='--',color='k')
	ax.set_xlim(6685,6735)
	ax.set_ylim(0.3,2.2)
	
	ax.set_xlabel('Wavelength ($\AA$)')
	ax.set_ylabel('Normalized Flux')

	plt.tight_layout()
	pp.savefig()
	
	fig,ax = plt.subplots(1)
	
	ax.plot(w_i,f_mcn_mcmc)
	ax.plot(w_i[cont_regions],f_mcn[cont_regions],'o',ms=3)
	ax.axvline(wi_start,ls=':',color='k')
	ax.axvline(wi_end,ls=':',color='k')
	
	ax.axvline(wi_start - (li*(2*vsig/c + 1.0)),ls=':',color='k',alpha=0.5)
	ax.axvline(wi_end + (li*(2*vsig/c + 1.0)),ls=':',color='k',alpha=0.5)
	
	ax.axvline(li,ls='--',color='k')
	ax.axhline(1.0,color='k')
	ax.set_xlim(6700,6720)
	ax.set_ylim(0.0,2.5)

	ax.set_xlabel('Wavelength ($\AA$)')
	ax.set_ylabel('Normalized Flux')
	
	plt.tight_layout()
	pp.savefig()
	
	fig,ax = plt.subplots(1)
	
	ax.hist(ew_dist,bins=25)
	ax.axvline(ew[2],ls='--',color='k')
	ax.axvline(ew[1],ls='--',color='k')
	ax.axvline(ew[0],ls='--',color='k')
	
	ax.set_xlabel('Li EW ($\AA$)')
	ax.set_ylabel('N')

	plt.tight_layout()
	pp.savefig()

	plt.close('all')

	pp.close()

	return


def region_select_pkl(target,template=None,tar_stretch=True, temp_stretch=True,reverse=False,dk_wav='wav',dk_flux='flux', tell_file=None,jump_to=0,reg_file=None):
	'''
	An interactive function to plot target and template spectra
	that allowing you to select useful regions with which to 
	compute the broadening functions, ccfs, ect.

	This funciton is meant for specrta in a pickled dictionary.

	For this function to work properly the template and target 
	spectrum have to have the same format, i.e. the same number 
	of orders and roughly the same wavelength coverage. 

	If a template spectrum is not specified, it will plot the 
	target twice, where it can be nice to have one strethed 
	and on not. 

	Functionality:
	The function brings up an interactive figure with the target 
	on top and the template on bottom. hitting the 'm' key will 
	mark wavelengths dotted red lines. The 'b' key will mark the 
	start of a region with a solid black line and then the end of 
	the region with a dashed black line. Regions should always go 
	from small wavelengths to larger wavelengths, and regions 
	should always close (.i.e., end with a dashed line). Hitting 
	the return key over the terminal will advance to the next order
	and it will print the region(s) you've created to the terminal
	screen that are in the format that the saphires.io.read 
	functions use. The regions from the previous order will show 
	up as dotted black lines allowing you to create regions that 
	do not overlap. 

	Parameters
	----------
	target : str
		File name for a pickled dictionary that has wavelength and 
		flux arrays for the target spectrum with the header keywords 
		defined in the dk_wav and dk_flux arguments.

	template : str, None
		File name for a pickled dictionary that has wavelength and 
		flux arrays for the target spectrum with the header keywords 
		defined in the dk_wav and dk_flux arguments. If None, the 
		target spectrum will be plotted in both panels. 

	tar_stretch : bool
		Option to window y-axis of the target spectrum plot on the 
		median with 50% above and below. This is useful for echelle 
		data with noisey edges. The default is True.

	temp_stretch ; bool
		Option to window y-axis of the template spectrum plot on the 
		median with 50% above and below. This is useful for echelle 
		data with noisey edges.The default is True.

	reverse : bool
		This function works best when the orders are ordered with
		ascending wavelength coverage. If this is not the case, 
		this option will flip them. The default is False, i.e., no 
		flip in the order.

	dk_wav : str
		Dictionary keyword for the wavelength array. Default is 'wav'

	dk_flux : str
		Dictionary keyword for the flux array. Default is 'flux'

	tell_file : optional keyword, None or str
		Name of file containing the location of telluric lines to be
		plotted as vertical lines. This is useful when selecting 
		regions free to telluric contamination. 
		File must be a tab/space separated ascii text file with the 
		following format:
		w_low w_high depth(compated the conintuum) w_central
		This is modeled after the MAKEE telluric template here:
		https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
		but just a heads up, these are in vaccum. 
		If None, this option is ignored. 
		The default is None. 

	jump_to : int
		Starting order. Useful when you want to pick up somewhere. 
		Default is 0.

	reg_file : optional keyword, None or str
		The name of a region file you want to overplay on the target and
		template spectra. The start of a regions will be a solid veritcal 
		grey line. The end will be a dahsed vertical grey line.
		The region file has the same formatting requirements as the io.read 
		functions. The default is None.

	Returns
	-------
	None

	'''
	l_range = []
	
	def press_key(event):
		if event.key == 'b':
			l_range.append(np.round(event.xdata,2))

			if (len(l_range)/2.0 % 1) != 0:
				ax[0].axvline(event.xdata,ls='-',color='k')
				ax[1].axvline(event.xdata,ls='-',color='k')
			else:
				ax[0].axvline(event.xdata,ls='--',color='k')
				ax[1].axvline(event.xdata,ls='--',color='k')
			plt.draw()

			return l_range

		if event.key == 'm':
			ax[0].axvline(event.xdata,ls=':',color='r')
			ax[1].axvline(event.xdata,ls=':',color='r')
			plt.draw()

			return

	#------ Reading in telluric file --------------

	if tell_file != None:
		wl,wh,r,w_tell = np.loadtxt(tell_file,unpack=True)

	#------ Reading in region file --------------

	if reg_file != None:
		name,reg_order,w_string = np.loadtxt(reg_file,unpack=True,dtype=nplts+'100,i,'+nplts+'1000')

	#----- Reading in and Formatiing ---------------	
	if template == None:
		template = copy.deepcopy(target)

	if py_version == 2:
		tar = pkl.load(open(target,'rb'))
		temp = pkl.load(open(template,'rb'))
	if py_version == 3:
		tar = pkl.load(open(target,'rb'),encoding='latin')
		temp = pkl.load(open(template,'rb'),encoding='latin')

	keys = list(tar.keys())

	if dk_wav not in keys:
		print("The wavelength array dictionary keyword specified, '"+dk_wav+"'")
		print("was not found.")
		return 0,0
	if dk_flux not in keys:
		print("The flux array dictionary keyword specified, '"+dk_flux+"'")
		print("was not found.")
		return 0,0

	if (tar[dk_wav].ndim == 1):
		order = 1

	if (tar[dk_wav].ndim > 1):
		order=tar[dk_wav].shape[0]

	if reverse == False:
		i = 0 + jump_to
	if reverse == True:
		i = order - jump_to - 1

	#-------------------------------------------------

	plt.ion()

	i = 0 + jump_to
	while i < order:
		if order > 1:
			if reverse == True:
				i_ind = order-1-i
			if reverse == False:
				i_ind = i
		
			flux = tar[dk_flux][i_ind]
			w = tar[dk_wav][i_ind]
			t_flux = temp[dk_flux][i_ind]
			t_w = temp[dk_wav][i_ind]

		else:
			i_ind = i
			flux = tar[dk_flux]
			w = tar[dk_wav]
			t_flux = temp[dk_flux]
			t_w = temp[dk_wav]

		#target
		w = w[~np.isnan(flux)]
		flux = flux[~np.isnan(flux)]

		w = w[np.isfinite(flux)]
		flux = flux[np.isfinite(flux)]

		#template
		t_w = t_w[~np.isnan(t_flux)]
		t_flux = t_flux[~np.isnan(t_flux)]

		t_w = t_w[np.isfinite(t_flux)]
		t_flux = t_flux[np.isfinite(t_flux)]
		
		#if ((w.size > 0) & (t_w.size > 0)):
		fig,ax=plt.subplots(2,sharex=True,figsize=(14.25,7.5))
	
		ax[0].set_title('Target - '+str(i_ind))
		ax[0].plot(w,flux)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[0].axvline(l_range[j],ls=':',color='red')
		ax[0].set_ylabel('Flux')
		ax[0].set_xlim(np.min(w),np.max(w))
		if tell_file != None:
			for j in range(w_tell.size):
				if ((w_tell[j] > np.min(w)) & (w_tell[j] < np.max(w))) == True:
					r_alpha = 1.0-r[j]
					ax[0].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
					ax[0].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
					ax[0].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)
					ax[1].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
					ax[1].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
					ax[1].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)

		if reg_file != None:
			if i_ind in reg_order:
				reg_ind = np.where(reg_order == i_ind)[0][0]
				n_regions=len(str(w_string[reg_ind]).split('-'))-1
				for j in range(n_regions):
					w_reg_start = float(w_string[reg_ind].split(',')[j].split('-')[0])
					w_reg_end = float(w_string[reg_ind].split(',')[j].split('-')[1])
					ax[0].axvline(w_reg_start,ls='-',color='grey')
					ax[0].axvline(w_reg_end,ls='--',color='grey')
					ax[1].axvline(w_reg_start,ls='-',color='grey')
					ax[1].axvline(w_reg_end,ls='--',color='grey')

		if tar_stretch == True:
			ax[0].axis([np.min(w),np.max(w),
		    	       np.median(flux)-np.median(flux)*0.5,
		        	   np.median(flux)+np.median(flux)*0.5])
		ax[0].grid(True,which='both',axis='both')
	
		ax[1].set_title('Template - '+str(i_ind))
		ax[1].plot(t_w,t_flux)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[1].axvline(l_range[j],ls=':',color='red')
		ax[1].set_ylabel('Flux')
		ax[1].set_xlabel('Wavelength')		

		if ((t_flux.size > 0)&(temp_stretch==True)):
			ax[1].axis([np.min(t_w),np.max(t_w),
			            np.median(t_flux)-np.median(t_flux)*0.5,
			            np.median(t_flux)+np.median(t_flux)*0.5])
		ax[1].grid(True,which='both',axis='both')
	
		plt.tight_layout()

		l_range = []
	
		cid = fig.canvas.mpl_connect('key_press_event',press_key)
	
		wait = p_input('')
	
		if wait != 'r':
			i = i+1
	
			if len(l_range) > 0:
				out_range=''
				for j in range(len(l_range)):
					if j < len(l_range)-1:
						if (j/2.0 % 1) != 0:
							out_range=out_range+str(l_range[j])+','
						if (j/2.0 % 1) == 0:
							out_range=out_range+str(l_range[j])+'-'
					if j == len(l_range)-1:
						out_range=out_range+str(l_range[j])
				print(target,i_ind,out_range)
	
			if (len(l_range) == 0) & (reg_file != None):
				if i_ind in reg_order:
					i_reg = np.where(reg_order == i_ind)[0][0]
					print(target,i_ind,w_string[i_reg])

		fig.canvas.mpl_disconnect(cid)
	
		plt.cla()
	
		plt.close()
        
	return


def region_select_vars(w,f,tar_stretch=True,reverse=False,tell_file=None,jump_to=0):
	'''
	An interactive function to plot spectra that allowing you 
	to select useful regions with which to compute the 
	broadening functions, ccfs, ect.

	Functionality:
	The function brings up an interactive figure with spectra. 
	Hitting the 'm' key will mark wavelengths with dotted red 
	lines. The 'b' key will mark the start of a region with a 
	solid black line and then the end of the region with a 
	dashed black line. Regions should always go from small 
	wavelengths to larger wavelengths, and regions should always 
	close (.i.e., end with a dashed line). Hitting the return 
	key over the terminal will advance to the next order and it 
	will print the region(s) you've created to the terminal
	screen that are in the format that the saphires.io.read_vars 
	function can use. The regions from the previous order will 
	show up as dotted black lines allowing you to create regions 
	that do not overlap. 

	Parameters
	----------
	w : array-like
		Wavelength array assumed to be in Angstroms.

	tar_stretch : bool
		Option to window y-axis of the spectrum plot on the 
		median with 50% above and below. This is useful for echelle 
		data with noisey edges. The default is True.

	reverse : bool
		This function works best when the orders are ordered with
		ascending wavelength coverage. If this is not the case, 
		this option will flip them. The default is False, i.e., no 
		flip in the order.

	tell_file : optional keyword, None or str
		Name of file containing the location of telluric lines to be
		plotted as vertical lines. This is useful when selecting 
		regions free to telluric contamination. 
		File must be a tab/space separated ascii text file with the 
		following format:
		w_low w_high depth(compated the conintuum) w_central
		This is modeled after the MAKEE telluric template here:
		https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
		but just a heads up, these are in vaccum. 
		If None, this option is ignored. 
		The default is None. 

	jump_to : int
		Starting order. Useful when you want to pick up somewhere. 
		Default is 0.

	Returns
	-------
	None

	'''
	l_range = []
	
	def press_key(event):
		if event.key == 'b':
			l_range.append(np.round(event.xdata,2))

			if (len(l_range)/2.0 % 1) != 0:
				ax[0].axvline(event.xdata,ls='-',color='k')
				ax[1].axvline(event.xdata,ls='-',color='k')
			else:
				ax[0].axvline(event.xdata,ls='--',color='k')
				ax[1].axvline(event.xdata,ls='--',color='k')
			plt.draw()

			return l_range

		if event.key == 'm':
			ax[0].axvline(event.xdata,ls=':',color='r')
			ax[1].axvline(event.xdata,ls=':',color='r')
			plt.draw()

			return

	#------ Reading in telluric file --------------

	if tell_file != None:
		wl,wh,r,w_tell = np.loadtxt(tell_file,unpack=True)

	#----- Reading in and Formatiing ---------------	
	if (w.ndim == 1):
		order = 1

	if (w.ndim > 1):
		order=w.shape[0]
	#-------------------------------------------------

	plt.ion()

	i = 0 + jump_to
	while i < order:
		if order > 1:
			if reverse == True:
				i_ind = order-1-i
			if reverse == False:
				i_ind = i
		
			flux_plot = f[i_ind]
			w_plot = w[i_ind]

		else:
			i_ind = i
			flux_plot = f
			w_plot = w

		#target
		w_plot = w_plot[~np.isnan(flux_plot)]
		flux_plot = flux_plot[~np.isnan(flux_plot)]

		w_plot = w_plot[np.isfinite(flux_plot)]
		flux_plot = flux_plot[np.isfinite(flux_plot)]

		fig,ax=plt.subplots(2,sharex=True,figsize=(14.25,7.5))

		ax[0].set_title('Target - '+str(i_ind))
		ax[0].plot(w_plot,flux_plot)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[0].axvline(l_range[j],ls=':',color='red')
		ax[0].set_ylabel('Flux')
		ax[0].set_xlim(np.min(w),np.max(w))
		if tell_file != None:
			for j in range(w_tell.size):
				if ((w_tell[j] > np.min(w)) & (w_tell[j] < np.max(w))) == True:
					r_alpha = 1.0-r[j]
					ax[0].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
					ax[0].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
					ax[0].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)
					ax[1].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
					ax[1].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
					ax[1].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)
		if tar_stretch == True:
			ax[0].axis([np.min(w_plot),np.max(w_plot),
		    	       np.median(flux_plot)-np.median(flux_plot)*0.5,
		        	   np.median(flux_plot)+np.median(flux_plot)*0.5])
		ax[0].grid(True,which='both',axis='both')

		ax[1].plot(w_plot,flux_plot)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[1].axvline(l_range[j],ls=':',color='red')
		ax[1].set_ylabel('Flux')
		ax[1].set_xlabel('Wavelength')		

		ax[1].grid(True,which='both',axis='both')

		plt.tight_layout()

		l_range = []

		cid = fig.canvas.mpl_connect('key_press_event',press_key)

		wait = p_input('')

		if wait != 'r':
			i = i+1

			if len(l_range) > 0:
				out_range=''
				for j in range(len(l_range)):
					if j < len(l_range)-1:
						if (j/2.0 % 1) != 0:
							out_range=out_range+str(l_range[j])+','
						if (j/2.0 % 1) == 0:
							out_range=out_range+str(l_range[j])+'-'
					if j == len(l_range)-1:
						out_range=out_range+str(l_range[j])
				print(i_ind,out_range)

		fig.canvas.mpl_disconnect(cid)

		plt.cla()

		plt.close()
        
	plt.ioff()

	return


def region_select_ms(target,template=None,tar_stretch=True, temp_stretch=True,reverse=False,t_order=0,temp_order=0,header_wave=False,w_order=1,temp_w_order=1,w_mult=1,igrins_default=False, tell_file=None,jump_to=0,reg_file=None):
	'''
	An interactive function to plot target and template spectra
	that allowing you to select useful regions with which to 
	compute the broadening functions, ccfs, ect.

	This funciton is meant for multi-order specrta in a fits file.

	For this function to work properly the template and target 
	spectrum have to have the same format, i.e. the same number 
	of orders and roughly the same wavelength coverage. 

	If a template spectrum is not specified, it will plot the 
	target twice, where it can be nice to have one strethed 
	and on not. 

	Functionality:
	The function brings up an interactive figure with the target 
	on top and the template on bottom. hitting the 'm' key will 
	mark wavelengths dotted red lines. The 'b' key will mark the 
	start of a region with a solid black line and then the end of 
	the region with a dashed black line. Regions should always go 
	from small wavelengths to larger wavelengths, and regions 
	should always close (.i.e., end with a dashed line). Hitting 
	the return key over the terminal will advance to the next order
	and it will print the region(s) you've created to the terminal
	screen that are in the format that the saphires.io.read 
	functions use (you may have to delete commas and parenthesis
	depending on whether you are running this command in python 2
	or 3). The regions from the previous order will show 
	up as dotted black lines allowing you to create regions that 
	do not overlap. 

	Parameters
	----------
	target : str
		File name for a pickled dictionary that has wavelength and 
		flux arrays for the target spectrum with the header keywords 
		defined in the dk_wav and dk_flux arguments.

	template : str, None
		File name for a pickled dictionary that has wavelength and 
		flux arrays for the target spectrum with the header keywords 
		defined in the dk_wav and dk_flux arguments. If None, the 
		target spectrum will be plotted in both panels. 

	tar_stretch : bool
		Option to window y-axis of the target spectrum plot on the 
		median with 50% above and below. This is useful for echelle 
		data with noisey edges. The default is True.

	temp_stretch ; bool
		Option to window y-axis of the template spectrum plot on the 
		median with 50% above and below. This is useful for echelle 
		data with noisey edges.The default is True.

	reverse : bool
		This function works best when the orders are ordered with
		ascending wavelength coverage. If this is not the case, 
		this option will flip them. The default is False, i.e., no 
		flip in the order.

	t_order : int
		The order of the target spectrum. Some multi-order spectra 
		come in multi-extension fits files (e.g. IGRINS). This 
		parameter defines that extension. The default is 0.

	temp_order : int
		The order of the template spectrum. Some multi-order spectra 
		come in multi-extension fits files (e.g. IGRINS). This 
		parameter defines that extension. The default is 0.

	w_order : int
		The order of the wavelength extension of the target spectrum. 
		Some multi-order spectra come in multi-extension fits files 
		(e.g. IGRINS). This parameter defines that extension. 
		The default is 1. This is the extension for the old IGRINS 
		data reduction format. In the updated version (v3), the extension
		is 3. 

	temp_w_order : int
		The order of the wavelength extension of the template spectrum. 
		Some multi-order spectra come in multi-extension fits files 
		(e.g. IGRINS). This parameter defines that extension. 
		The default is 1. This is the extension for the old IGRINS 
		data reduction format. In the updated version (v3), the extension
		is 3. 

	header_wave : bool or 'Single'
		Whether to assign the wavelength array from the header keywords or
		from a separate fits extension. If True, it uses the header keywords,
		assumiing they are linearly spaced. If False, it looks in the second 
		fits extension, i.e. hdu[1].data
		If header_wave is set to 'Single', it treats each fits extension like
		single order fits file that could be read in with saph.io.read_fits. 
		This feature is useful for SALT/HRS specrtra reduced with the MIDAS 
		pipeline.

	w_mult : float
		Value to multiply the wavelength array. This is used to convert the 
		input wavelength array to Angstroms if it is not already. The default 
		is 1, assuming the wavelength array is alreay in Angstroms. 

	igrins_default : bool
		The option to override all of the input arguments to parameters
		that are tailored to IGRINS data. Keyword arguments will be set 
		to:
		t_order = 0
		temp_order = 3
		temp_stretch = False
		header_wave = True
		w_mult = 10**4
		reverse = True

	tell_file : optional keyword, None or str
		Name of file containing the location of telluric lines to be
		plotted as vertical lines. This is useful when selecting 
		regions free to telluric contamination. 
		File must be a tab/space separated ascii text file with the 
		following format:
		w_low w_high depth(compated the conintuum) w_central
		This is modeled after the MAKEE telluric template here:
		https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt
		but just a heads up, these are in vaccum. 
		If None, this option is ignored. 
		The default is None. 

	jump_to : int
		Starting order. Useful when you want to pick up somewhere. 
		Default is 0.

	reg_file : optional keyword, None or str
		The name of a region file you want to overplay on the target and
		template spectra. The start of a regions will be a solid veritcal 
		grey line. The end will be a dahsed vertical grey line.
		The region file has the same formatting requirements as the io.read 
		functions. The default is None.

	Returns
	-------
	None

	'''
	l_range = []
	
	def press_key(event):
		if event.key == 'b':
			l_range.append(np.round(event.xdata,2))

			if (len(l_range)/2.0 % 1) != 0:
				ax[0].axvline(event.xdata,ls='-',color='k')
				ax[1].axvline(event.xdata,ls='-',color='k')
			else:
				ax[0].axvline(event.xdata,ls='--',color='k')
				ax[1].axvline(event.xdata,ls='--',color='k')
			plt.draw()

			return l_range

		if event.key == 'm':
			ax[0].axvline(event.xdata,ls=':',color='r')
			ax[1].axvline(event.xdata,ls=':',color='r')
			plt.draw()

			return

	if igrins_default == True:
		t_order = 0
		temp_order = 3
		temp_stretch = False
		header_wave = False
		w_mult = 10**4
		reverse = False

	#------ Reading in telluric file --------------

	if tell_file != None:
		wl,wh,r,w_tell = np.loadtxt(tell_file,unpack=True)

	#------ Reading in region file --------------

	if reg_file != None:
		name,reg_order,w_string = np.loadtxt(reg_file,unpack=True,dtype=nplts+'100,i,'+nplts+'1000')
		if (name.size == 1): 
			name=np.array([name])
			reg_order=np.array([reg_order])
			w_string=np.array([w_string])

	#----- Reading in and Formatiing ---------------	
	if template == None:
		template = copy.deepcopy(target)

	hdulist = pyfits.open(target)
	t_hdulist = pyfits.open(template)

	if header_wave != 'Single':
		order = hdulist[t_order].data.shape[0]
	else:
		order = len(hdulist)
		if order == 1:
			order = hdulist[0].data.shape[0]

	plt.ion()

	if reverse == False:
		i = 0 + jump_to
	if reverse == True:
		i = order - jump_to - 1

	while i < order:
		if reverse == True:
			i_ind = order-1-i
		if reverse == False:
			i_ind = i

		#---- Read in the Target -----
		if header_wave == 'Single':
			flux=hdulist[i_ind].data
			w0=float(hdulist[i_ind].header['CRVAL1'])
			dw=float(hdulist[i_ind].header['CDELT1'])

			if 'LTV1' in hdulist[i_ind].header:
				shift=float(hdulist[i_ind].header['LTV1'])
				w0=w0-shift*dw

			w0=w0 * w_mult
			dw=dw * w_mult

			w=np.arange(flux.size)*dw+w0

		if header_wave == False:
			flux = hdulist[t_order].data[i_ind]
			
			w = hdulist[w_order].data[i_ind]*w_mult
			dw=(np.max(w) - np.min(w))/float(w.size)

		print(header_wave)
		print(order,t_order,i_ind)


		if header_wave == True:
			try:
				flux = hdulist[order].data[i_ind]
			except:
				flux = hdulist[t_order].data[i_ind]

			try:
				#Pulls out all headers that have the WAT2 keywords
				header_keys=np.array(hdulist[t_order].header.keys(),dtype=str)
				header_test=np.array([header_keys[d][0:4]=='WAT2' \
				                     for d in range(header_keys.size)])
				w_sol_inds=np.where(header_test==True)[0]
	
				#The loop below puts all the header extensions into one string
				w_sol_str=''
				for j in range(w_sol_inds.size):
				    if len(hdulist[t_order].header[w_sol_inds[j]]) == 68:
				        w_sol_str=w_sol_str+hdulist[t_order].header[w_sol_inds[j]]
				    if len(hdulist[t_order].header[w_sol_inds[j]]) == 67:
				        w_sol_str=w_sol_str+hdulist[t_order].header[w_sol_inds[j]]+' '
				    if len(hdulist[t_order].header[w_sol_inds[j]]) == 66:
				        w_sol_str=w_sol_str+hdulist[t_order].header[w_sol_inds[j]]+' ' 
				    if len(hdulist[t_order].header[w_sol_inds[j]]) < 66:
				        w_sol_str=w_sol_str+hdulist[t_order].header[w_sol_inds[j]]

			except: 
				wat2_head = hdulist[t_order].header['WAT2*']
				w_sol_inds = np.arange(len(wat2_head))

				w_sol_str=''
				for j in range(w_sol_inds.size):
				    if len(wat2_head[int(w_sol_inds[j])]) == 68:
				        w_sol_str=w_sol_str+wat2_head[int(w_sol_inds[j])]
				    if len(wat2_head[int(w_sol_inds[j])]) == 67:
				        w_sol_str=w_sol_str+wat2_head[int(w_sol_inds[j])]+' '
				    if len(wat2_head[int(w_sol_inds[j])]) == 66:
				        w_sol_str=w_sol_str+wat2_head[int(w_sol_inds[j])]+' ' 
				    if len(wat2_head[int(w_sol_inds[j])]) < 66:
				        w_sol_str=w_sol_str+wat2_head[int(w_sol_inds[j])]

			# normalized the formatting
			w_sol_str=w_sol_str.replace('    ',' ').replace('   ',' ').replace('  ',' ')

			# removed wavelength solution preamble
			w_sol_str=w_sol_str[16:]

			#Check that the wavelength solution is linear.
			w_parameters = len(w_sol_str.split(' = ')[1].split(' '))
			if w_parameters > 11:
				print('Your header wavelength solution is not linear')
				print('Non-linear wavelength solutions are not currently supported')
				print('Aborting...')
				return 

			w_type = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[3])
			if w_type != 0:
				print('Your header wavelength solution is not linear')
				print('Non-linear wavelength solutions are not currently supported')
				print('Aborting...')
				return 
				
			w0 = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[5])
			dw = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[6])
			z = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[7])

			w = ((np.arange(flux.size)*dw+w0)/(1+z))*w_mult


		#---- Read in the Template -----
		if header_wave == 'Single':
			t_flux=t_hdulist[i_ind].data
			t_w0=float(t_hdulist[i_ind].header['CRVAL1'])
			t_dw=float(t_hdulist[i_ind].header['CDELT1'])

			if 'LTV1' in t_hdulist[i_ind].header:
				t_shift=float(t_hdulist[i_ind].header['LTV1'])
				t_w0=t_w0-t_shift*t_dw

			t_w0=t_w0 * w_mult
			t_dw=t_dw * w_mult

			t_w=np.arange(t_flux.size)*t_dw+t_w0

		if header_wave == False:
			t_flux = t_hdulist[temp_order].data[i_ind]
			
			t_w = t_hdulist[temp_w_order].data[i_ind]*w_mult
			t_dw=(np.max(t_w) - np.min(t_w))/float(t_w.size)

		if header_wave == True:
			t_flux = t_hdulist[temp_order].data[i_ind]

			#Pulls out all headers that have the WAT2 keywords
			header_keys=np.array(t_hdulist[temp_order].header.keys(),dtype=str)
			header_test=np.array([header_keys[d][0:4]=='WAT2' \
			                     for d in range(header_keys.size)])
			w_sol_inds=np.where(header_test==True)[0]

			#The loop below puts all the header extensions into one string
			w_sol_str=''
			for j in range(w_sol_inds.size):
			    if len(t_hdulist[temp_order].header[w_sol_inds[j]]) == 68:
			        w_sol_str=w_sol_str+t_hdulist[temp_order].header[w_sol_inds[j]]
			    if len(t_hdulist[temp_order].header[w_sol_inds[j]]) == 67:
			        w_sol_str=w_sol_str+t_hdulist[temp_order].header[w_sol_inds[j]]+' '
			    if len(t_hdulist[temp_order].header[w_sol_inds[j]]) == 66:
			        w_sol_str=w_sol_str+t_hdulist[temp_order].header[w_sol_inds[j]]+' ' 
			    if len(t_hdulist[temp_order].header[w_sol_inds[j]]) < 66:
			        w_sol_str=w_sol_str+t_hdulist[temp_order].header[w_sol_inds[j]]

			# normalized the formatting
			w_sol_str=w_sol_str.replace('    ',' ').replace('   ',' ').replace('  ',' ')

			# removed wavelength solution preamble
			w_sol_str=w_sol_str[16:]

			#Check that the wavelength solution is linear.
			w_parameters = len(w_sol_str.split(' = ')[1].split(' '))
			if w_parameters > 11:
				print('Your header wavelength solution is not linear')
				print('Non-linear wavelength solutions are not currently supported')
				print('Aborting...')
				return 

			w_type = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[3])
			if w_type != 0:
				print('Your header wavelength solution is not linear')
				print('Non-linear wavelength solutions are not currently supported')
				print('Aborting...')
				return 
				
			t_w0 = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[5])
			t_dw = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[6])
			z = float(w_sol_str.split('spec')[1:][order[i]].split(' ')[7])

			t_w = ((np.arange(t_flux.size)*t_dw+t_w0)/(1+z))*w_mult

		#target
		w = w[~np.isnan(flux)]
		flux = flux[~np.isnan(flux)]

		w = w[np.isfinite(flux)]
		flux = flux[np.isfinite(flux)]

		#template
		t_w = t_w[~np.isnan(t_flux)]
		t_flux = t_flux[~np.isnan(t_flux)]

		t_w = t_w[np.isfinite(t_flux)]
		t_flux = t_flux[np.isfinite(t_flux)]
		
		
		#--------Interactive Plotting --------
		fig,ax=plt.subplots(2,sharex=True,figsize=(14.25,7.5))
	
		if ((w.size > 0) &(t_w.size > 0)):

			ax[0].set_title('Target - '+str(i_ind))
			ax[0].plot(w,flux)
			if len(l_range) > 0:
				for j in range(len(l_range)):
					ax[0].axvline(l_range[j],ls=':',color='red')
			ax[0].set_ylabel('Flux')
			ax[0].set_xlim(np.min(w),np.max(w))
			if tell_file != None:
				for j in range(w_tell.size):
					if ((w_tell[j] > np.min(w)) & (w_tell[j] < np.max(w))) == True:
						r_alpha = 1.0-r[j]
						ax[0].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
						ax[0].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
						ax[0].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)
						ax[1].axvline(w_tell[j],ls='--',color='blue',alpha=r_alpha)
						ax[1].axvline(wl[j],ls=':',color='blue',alpha=r_alpha)
						ax[1].axvline(wh[j],ls=':',color='blue',alpha=r_alpha)

			if reg_file != None:
				if i_ind in reg_order:
					i_reg = np.where(reg_order == i_ind)[0][0]
					n_regions=len(str(w_string[i_reg]).split('-'))-1
					for j in range(n_regions):
						w_reg_start = float(w_string[i_reg].split(',')[j].split('-')[0])
						w_reg_end = float(w_string[i_reg].split(',')[j].split('-')[1])
						ax[0].axvline(w_reg_start,ls='-',color='grey')
						ax[0].axvline(w_reg_end,ls='--',color='grey')
						ax[1].axvline(w_reg_start,ls='-',color='grey')
						ax[1].axvline(w_reg_end,ls='--',color='grey')

			if tar_stretch == True:
				ax[0].axis([np.min(w),np.max(w),
			    	       np.median(flux)-np.median(flux)*0.5,
			        	   np.median(flux)+np.median(flux)*0.5])
			ax[0].grid(True,which='both',axis='both')
	
			ax[1].set_title('Template - '+str(i_ind))
			ax[1].plot(t_w,t_flux)
			if len(l_range) > 0:
				for j in range(len(l_range)):
					ax[1].axvline(l_range[j],ls=':',color='red')
			ax[1].set_ylabel('Flux')
			ax[1].set_xlabel('Wavelength')		
			if ((t_flux.size > 0)&(temp_stretch==True)):
				ax[1].axis([np.min(t_w),np.max(t_w),
				            np.median(t_flux)-np.median(t_flux)*0.5,
				            np.median(t_flux)+np.median(t_flux)*0.5])
			ax[1].grid(True,which='both',axis='both')
	
			plt.tight_layout()
	
			l_range = []
	
			cid = fig.canvas.mpl_connect('key_press_event',press_key)
	
		wait = p_input('')
	
		if wait != 'r':
			i = i+1
	
			if len(l_range) > 0:
				out_range=''
				for j in range(len(l_range)):
					if j < len(l_range)-1:
						if (j/2.0 % 1) != 0:
							out_range=out_range+str(l_range[j])+','
						if (j/2.0 % 1) == 0:
							out_range=out_range+str(l_range[j])+'-'
					if j == len(l_range)-1:
						out_range=out_range+str(l_range[j])
				print(target,i_ind,out_range)
			if (len(l_range) == 0) & (reg_file != None):
				if i_ind in reg_order:
					i_reg = np.where(reg_order == i_ind)[0][0]
					print(target,i_ind,w_string[i_reg])
	
		fig.canvas.mpl_disconnect(cid)
	
		plt.cla()
	
		plt.close()
        
	return


def sex2dd(ra,dec,results=False):
    '''
    Convert ra and dec in sexigesimal format to decimal degrees.

    Parameters:
    -----------
    ra : ndarray; str
        String sexigesimal RAs in the format 'HH:MM:SS.SS'. Can be an array of
        RAs.

    dec : ndarray; str
        String sexigesimal Dec in format '+/-DD:MM:SS.SS'. Can be an array of
        Decs.

    Results : bool
        If True, the decimal degree RA and Dec results are printed. 

    Returns:
    --------
    radd : ndarray
        Array of RAs in decimal degrees.

    decdd : ndarray
        Array of Decs in decimal degrees. 

    Output:
    -------
    N/A

    Version History:
    ----------------
    2015-05-15 - Start
    2016-12-01 - Added the one-line for-loops that split up the strings so the 
                 function could actually handle an array rather than single 
                 values. 
    '''
    
    
    if type(ra) is np.ndarray:
        rah=np.array([float(ra[d][0:2]) for d in range(ra.size)])
        ram=np.array([float(ra[d][3:5]) for d in range(ra.size)])
        ras=np.array([float(ra[d][6:]) for d in range(ra.size)])
        decd=np.array([float(dec[d][0:3]) for d in range(ra.size)])
        decm=np.array([float(dec[d][4:6]) for d in range(ra.size)])
        decs=np.array([float(dec[d][7:]) for d in range(ra.size)])      
        radd=((rah+ram/60.0+ras/3600.0)/24.0)*360.0
        for i in range(decd.size):
            if dec[i][0] == '+':
                decdd=(decd+decm/60.0+decs/3600.0)
            else:
                decdd=-(np.abs(decd)+decm/60.0+decs/3600.0)

    if type(ra) is str:
        rah=np.array(float(ra[0:2]))
        ram=np.array(float(ra[3:5]))
        ras=np.array(float(ra[6:]))
        decd=np.array(float(dec[0:3]))
        decm=np.array(float(dec[4:6]))
        decs=np.array(float(dec[7:]))
        radd=((rah+ram/60.0+ras/3600.0)/24.0)*360.0
        if dec[0] == '+':
            decdd=(decd+decm/60.0+decs/3600.0)
        else:
            decdd=-(np.abs(decd)+decm/60.0+decs/3600.0)

    if results == True:
        print(radd,decdd)

    return radd,decdd


def sigma_clip(data, sig=3, iters=1, cenfunc=np.median, varfunc=np.var,maout=False):
    """ 
    Perform sigma-clipping on the provided data.

    This performs the sigma clipping algorithm - i.e. the data will be iterated
    over, each time rejecting points that are more than a specified number of
    standard deviations discrepant.

    .. note::
        `scipy.stats.sigmaclip` provides a subset of the functionality in this
        function.

    Parameters
    ----------
    data : array-like
        The data to be sigma-clipped (any shape).
    sig : float
        The number of standard deviations (*not* variances) to use as the
        clipping limit.
    iters : int or None
        The number of iterations to perform clipping for, or None to clip until
        convergence is achieved (i.e. continue until the last iteration clips
        nothing).
    cenfunc : callable
        The technique to compute the center for the clipping. Must be a
        callable that takes in a 1D data array and outputs the central value.
        Defaults to the median.
    varfunc : callable
        The technique to compute the variance about the center. Must be a
        callable that takes in a 1D data array and outputs the width estimator
        that will be interpreted as a variance. Defaults to the variance.
    maout : bool or 'copy'
        If True, a masked array will be returned. If the special string
        'inplace', the masked array will contain the same array as `data`,
        otherwise the array data will be copied.

    Returns
    -------
    filtereddata : `numpy.ndarray` or `numpy.masked.MaskedArray`
        If `maout` is True, this is a masked array with a shape matching the
        input that is masked where the algorithm has rejected those values.
        Otherwise, a 1D array of values including only those that are not
        clipped.
    mask : boolean array
        Only present if `maout` is False. A boolean array with a shape matching
        the input `data` that is False for rejected values and True for all
        others.

    Examples
    --------
    This will generate random variates from a Gaussian distribution and return
    only the points that are within 2 *sample* standard deviation from the
    median::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> data,mask = sigma_clip(randvar, 2, 1)

    This will clipping on a similar distribution, but for 3 sigma relative to
    the sample *mean*, will clip until converged, and produces a
    `numpy.masked.MaskedArray`::

        >>> from astropy.stats import sigma_clip
        >>> from numpy.random import randn
        >>> from numpy import mean
        >>> randvar = randn(10000)
        >>> maskedarr = sigma_clip(randvar, 3, None, mean, maout=True)

    """

    data = np.array(data, copy=False)
    oldshape = data.shape
    data = data.ravel()

    mask = np.ones(data.size, bool)
    if iters is None:
        i = -1
        lastrej = sum(mask) + 1
        while(sum(mask) != lastrej):
            i += 1
            lastrej = sum(mask)
            do = data - cenfunc(data[mask])
            mask = do * do <= varfunc(data[mask]) * sig ** 2
        iters = i + 1
        #TODO: ?print iters to the log if iters was None?
    else:
        for i in range(iters):
            do = data - cenfunc(data[mask])
            mask = do * do <= varfunc(data[mask]) * sig ** 2

    if maout:
        return np.ma.MaskedArray(data, ~mask, copy=maout != 'inplace')
    else:
        return data[mask], mask.reshape(oldshape)


def spec_trim(w_tar,f,w_range,temp_trim,trim_style='clip'):
	'''
	A function to select certain regions of a spectrum with which
	to compute the broadedning function. 

	trim_style - refers to how you want to deal with the bad regions
	- 'clip' - remove data all together. This creates edges that can 
			   cause noise in the BF
	- 'lin'  - linearly interpolates over clipped regions
	- 'spl'  - interpolated over the clipped regions with a cubic 
			   spline - don't use this option.
	
	Paramters
	---------
	w_tar : array-like
		Wavelength array, must be one dimensional.
	
	f : array-like
		Flux array, must be one dimensional.
	
	w_range : str
		Wavelength trimming string for the target star. 
		Must have the general form "w1-w2,w3-w4" where '-' 
		symbols includes the wavelength region, ',' symbols 
		excludes them. There can be as many regions as you 
		want, as long as it ends with an inclusive region 
		(i.e. cannot end with a comma or dash). Wavelength 
		values must ascend left to right. The '*' symbol 
		includes everything.

	temp_trim : str
		Wavelength trimming string for the template star. 
		Must have the general form "w1-w2,w3-w4" where '-' 
		symbols includes the wavelength region, ',' symbols 
		excludes them. There can be as many regions as you 
		want, as long as it ends with an inclusive region 
		(i.e. cannot end with a comma or dash). Wavelength 
		values must ascend left to right. The '*' symbol 
		includes everything.
		

	trim_style : str, options: 'clip', 'lin', 'spl'
		If a wavelength region file is input in the 'spectra_list' parameter, 
		this parameter describes how gaps are dealt with. 
		- If 'clip', unused regions will be left as gaps.
		- If 'lin', unused regions will be linearly interpolated over.
		- If 'spl', unused regions will be interpolated over with a cubic 
		  spline. You probably don't want to use this one.

	Returns
	-------
	w_tar : str
		The trimmed wavelength array

	f : str
		The trimmed flux array
	'''
	
	#The following makes the indexing selection array, t_ind under various 
	#conditional situtations
	if w_range != '*':
		t_ind = np.zeros(w_tar.size,dtype=bool)

		n_regions = len(str(w_range).split(','))

		for j in range(n_regions):
			w_start=str(w_range).split(',')[j].split('-')[0]
			w_end=str(w_range).split(',')[j].split('-')[1]
			
			t_ind[(w_tar >= float(w_start)) & (w_tar <= float(w_end))] = True

		if temp_trim != '*':

			n_regions=len(str(temp_trim).split('-'))

			for j in range(n_regions+1):
				if j == 0:
					w_start=str(temp_trim).split('-')[j]
					if np.min(w_tar) < float(w_start):
						t_ind[w_tar <= float(w_start)] = False

				if (j > 0) & (j < n_regions-1):
					i_range=str(temp_trim).split('-')[j]
					w_start=i_range.split(',')[0]
					w_end=i_range.split(',')[1]
					if (np.min(w_tar) > float(w_start)) & (np.min(w_tar) < float(w_end)):
						t_ind[w_tar < float(w_end)] = False

					if (np.min(w_tar) < float(w_start)) & (np.max(w_tar) > float(w_end)):
						t_ind[(w_tar > float(w_start)) & 
								(w_tar < float(w_end))] = False

					if (np.max(w_tar) > float(w_start)) & (np.max(w_tar) < float(w_end)):
						t_ind[w_tar > float(w_start)] = False

				if j == n_regions:
					w_end=str(temp_trim).split('-')[-1]
					if (np.max(w_tar) < float(w_end)):
						t_ind[w_tar > float(w_end)] = False
					if (np.min(w_tar) > float(w_end)):
						t_ind[w_tar > float(w_end)] = False

	if ((temp_trim != '*') & (w_range == '*')):

		t_ind = np.zeros(w_tar.size,dtype=bool)

		n_regions = len(str(temp_trim).split(','))

		for j in range(n_regions):
			w_start=str(temp_trim).split(',')[j].split('-')[0]
			w_end=str(temp_trim).split(',')[j].split('-')[1]
			t_ind[(w_tar >= float(w_start)) & 
				(w_tar <= float(w_end))] = True

	if ((w_range == '*') & (temp_trim == '*')):
		t_ind = np.ones(w_tar.size,dtype=bool)

	#This part deal with how you want to handle the specified regions:
	if trim_style == 'clip':
		w_tar = w_tar[t_ind]
		f = f[t_ind]

	if trim_style == 'lin':
		if w_tar[t_ind].size == 0:
			w_tar = w_tar[t_ind]
			f = f[t_ind]
			return w_tar,f

		if w_tar[t_ind].size > 0:
			f_lin = interpolate.interp1d(w_tar[t_ind],f[t_ind])
			w_tar = w_tar[(w_tar >= np.min(w_tar[t_ind])) & (w_tar <= np.max(w_tar[t_ind]))]
			f = f_lin(w_tar)
		
	if trim_style == 'spl':
		if w_tar[t_ind].size == 0:
			w_tar = w_tar[t_ind]
			f = f[t_ind]
			return w_tar,f
			
		if w_tar[t_ind].size > 0:
			f_cs = interpolate.CubicSpline(w_tar[t_ind],f[t_ind])
			w_tar = w_tar[(w_tar >= np.min(w_tar[t_ind])) & (w_tar <= np.max(w_tar[t_ind]))]
			f = f_cs(w_tar)

	return w_tar,f


def spec_ccf(f_s,f_t,m,v_spacing):
	'''
	This is the "under the hood" cross correlation function
	called by xc.todcor and xc.ccf.

	If you are looking for a ccf that plays nice with the
	SAPHIRES dictionaties, use xc.ccf.

	CCF Specifics: In this implementation, the cross correlation
	at each point is normalized by the number of flux array values 
	that went into that point, NOT the total number of point in the 
	array. This makes the most sense to me, but most formulae you 
	find do not do this.

	Parameters
	----------
	f_s: array-like
		Input flux array from the science spectrum. Array assumes
		that the corresponding wavelength array is spaced 
		logarithmicly, i.e. in linear velocity spacing, and are 
		continuum normalized and inverted. 

	f_t: array-like
		Input flux array from the template spectrum. Array assumes
		that the corresponding wavelength array is spaced 
		logarithmicly, i.e. in linear velocity spacing, and are 
		continuum normalized and inverted. 

	m : int
		Number of units in velocity space with which to compute the 
		cross correlation, must be an odd number. 

	v_spacing : float
		The velocity spacing of the input flux arrays -- must be the 
		same between them.

	Returns
	-------
	ccf : array-like
		The computed cross correlation function

	ccf_v : array-like
		The velcoity that corresponds to each cross correaltion value
		above. 

	'''
	if (m/2.0 % 1) == 0:
		m=m-1
		print('Subtracting 1 from m because it is even.')

	f_s_ccf = f_s / np.std(f_s)
	f_s_ccf = f_s_ccf - np.mean(f_s_ccf)

	f_t_ccf = f_t / np.std(f_t)
	f_t_ccf = f_t_ccf - np.mean(f_t_ccf)

	ccf = np.zeros(m)
	for i in range(m):
		ccf_i = i-(m-1)//2
		if ccf_i < 0:
			ccf[i] = (np.sum( f_s_ccf[:ccf_i] * 
			                 np.roll(f_t_ccf,ccf_i)[:ccf_i] ) / 
			          float(f_s_ccf[:ccf_i].size))
		if ccf_i >=0:
			ccf[i] = (np.sum( f_s_ccf[ccf_i:] * 
			                 np.roll(f_t_ccf,ccf_i)[ccf_i:] ) / 
					  float(f_s_ccf[ccf_i:].size))

	ccf_v = (np.arange(ccf.size)-(ccf.size-1)//2)*v_spacing

	return ccf,ccf_v


def td_gaussian(xy_ins, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
	'''
	A two-dimensional gaussian fuction, that is tailored to fitting data.

	Parameters
	----------
	xy_ins : array-like
		A two dimensional array of the input x and y values. I usually input
		something that has been through np.meshgrid, e.g., 
		x_in,y_in = np.meshgrid(x,y)
	
	amplitude : float
		Amplitude of the 2D gaussian.

	xo : float
		Center of the 2D gaussian along the x axis.

	yo : float
		Center of the 2D gaussian along the y axis.

	sigma_x : float
		Width of the 2D gaussian along the x axis.

	sigma_y : float
		Width of the 2D gaussian along the y axis.

	theta : float
		Position angle of the 2D gaussian.

	offset : float 
		Veritcal offset (in the z direction) of the 2D gaussian.

	Returns
	-------
	z : array-like
		A "raveled" array of the veritcal points. 
		Here is how you unravel it:
			x_in,y_in = np.meshgrid(x,y)
			gauss2d = td_gaussian((x_in,y_in),amplitude,xo,yo,sigma_x,sigma_y,
									  theta,offset)
			gauss = fit_gauss2d.reshape(x.size,y.size)
		Now you can plot it like:
			ax.contour(x,y,gauss)

	'''
	x,y = xy_ins
	xo = float(xo)
	yo = float(yo)    
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

	z = g.ravel()

	return z


def t_gaussian_off(x,A1,x01,sig1,A2,x02,sig2,A3,x03,sig3,o):
    '''
    A triple gaussian function with a constant vetical offset.

    Parameters
	----------
	x : array-like
		Array of x values over which the Gaussian profile will 
		be computed.

	A1 : float
		Amplitude of the first Gaussian profile. 

	x01 : float
		Center of the first Gaussian profile.

	sig1 : float
		Standard deviation (sigma) of the first Gaussian profile. 

	A2 : float
		Amplitude of the second Gaussian profile. 

	x02 : float
		Center of the second Gaussian profile.

	sig2 : float
		Standard deviation (sigma) of the second Gaussian profile. 

	A3 : float
		Amplitude of the third Gaussian profile. 

	x03 : float
		Center of the third Gaussian profile.

	sig3 : float
		Standard deviation (sigma) of the third Gaussian profile. 

	o : float
		Vertical offset of the Gaussian mixture. 

    Returns
	-------
	profile : array-like
		The Gaussian mixture specified over the input x array.
		Array has the same length as x.
    '''
    return (A1*np.e**(-(x-x01)**2/(2.0*sig1**2))+
            A2*np.e**(-(x-x02)**2/(2.0*sig2**2))+
            A3*np.e**(-(x-x03)**2/(2.0*sig3**2))+o)


def trap_int(x,y,yerr=None,xerr=None,y_low=None,y_lowerr=None):
    '''
    A function to integrate data with trapazoids. Returns the numerical 
    integration and the uncertainty in that integration. Can be integtrated 
    with respect to zero or to another curve sampled at the same rate. 

    Parameters
    ----------
    x : array like
       An array of x-values to be integrated over. 

    y : array like
       An array of y-values to be integrated. 

    yerr : array like, optional
       An array of one-sigma errors associated with y. Assumes no error of left
       as None, which is the default. 

    xerr : array like, optional
       An array of one-sigma errors associated with x. Assumes no error of left
       as None, which is the default. 

    y_low : array like, optional
       An array of y-values the y values above should be integrated against, 
       i.e. y-y_low. This array must have the same number of elements as y, and
       be interpolated to match the x values associated with y. If unassigned,
       (left as None) the integration is calculated with repsect to zero, i.e.
       y-0.0.
    
    y_lowerr : array like optional
       An array of one-sigma error associated with y_low. Only used if y_low is
       specified. 

    Returns
    -------
    Two values:
       The integration value and its error. 

    Outputs
    -------
    None

    Version History
    ---------------
    2016-12-06 - Start
    '''
    int_step=np.empty(0)
    int_step_err=np.empty(0)

    if y_lowerr == None: y_lowerr=np.zeros(y.size)
    if y_low == None:
        y_low=np.zeros(y.size)
        y_lowerr=np.zeros(y.size)
    if yerr is None: yerr=np.zeros(y.size)
    if xerr is None: xerr=np.zeros(y.size)

    for i in range(y.size-1):
        w=(x[i+1]-x[i])
        werr=np.sqrt(xerr[i+1]**2+xerr[i]**2)
        a=y[i]-y_low[i]
        aerr=np.sqrt(yerr[i]**2+y_lowerr[i]**2)
        b=y[i+1]-y_low[i+1]
        berr=np.sqrt(yerr[i+1]**2+y_lowerr[i+1]**2)
        int_step=np.append(int_step,(a+b)*w/2.0)
        int_step_err=np.append(int_step_err,np.sqrt((w*aerr/2.0)**2 + 
                                                    (w*berr/2.0)**2 + 
                                                    ((a+b)*werr/2.0)**2))
        
    return np.sum(int_step),np.sqrt(np.sum(int_step_err**2))


def w2linevel(data,w_line,v_width,bvcorr,gamma,cn = False):
	'''
	A function to take in a spectrum in wavelength space and 
	return a velocity spectrum of a specific line. 

	Parameters
    ----------
    data : tuple or string
    	If a tuple, data should have the form (w,f) where w is the wavelength array
    	and f is the flux array. It is assumed that these arrays are one dimensional 
    	and that they contain the line of iterest. 
    	If a string, data can be the name of a saphires-friendly dictionary in pickle
    	format, where the wavelength keyword is 'wav' and the flux keyword is 'flux'. 
    	The function will read in the pickle and find the order the line resides in.

    w_line : str or float
    	The wavelength of the line you want to see the velocity structure of.
    	There is a library of lines saved that you can call with a string. They are:
    		'ha': 6562.79
    		'hb': 4861.35
    		'hg': 4340.472
    		'hd': 4101.734
    		'hep': 3970.075
    		'hei': 5875.64
    		'li': 6707.81

	v_width : float
		The width in velocity space (km/s) you want the output velocity spectrum in.
		A value of 300 will return +/- 300 km/s centered on the line of interest.

	bvcorr : float
		The barycentric RV correction in km/s for the spectrum. It's shift if removed.
		Leave as 0 if the spectrum is alreay corrected for barycentric motion.

	gamma : float
		The center of mass velocity for the object. It's shift is removed. 

	cn : bool (optional)
		Option to continuum normalize the input spectrum.
		Current normalization parameters are:
			w_width=200.0
			maxiter=15
			nord=3
			lower=0.5
			upper=0.5
		A future version may make these parameters optional inputs, they may not be 
		ideal for your spectrum

    Returns
    -------
	vel : array

	flux : array

    Outputs
    -------
    None

    Version History
    ---------------
    2016-12-06 - Start
	'''
	line_options = ['ha','hb','hg','hd','hep','hei','li']

	if isinstance(w_line,str):
		if w_line in line_options:
			if w_line == 'ha':
				w_line = 6562.79
			if w_line == 'hb':
				w_line = 4861.35
			if w_line == 'hg':
				w_line = 4340.472
			if w_line == 'hd':
				w_line = 4101.734
			if w_line == 'hep':
				w_line = 3970.075
			if w_line == 'hei':
				w_line = 5875.64
			if w_line == 'li':
				w_line = 6707.81
		else:
			print('The line sting you provided is not in the library.')
			print('Options are:',line_options)
			print('Enter your own wavelength in angstroms, or update the library.')
	
			return 0,0

	if isinstance(data,tuple):
		w = data[0]
		f = data[1]

	if isinstance(data,str):
		spec = pkl.load(open(data,'rb'))
		if len(spec['wav'].shape) > 1:
			dw = np.zeros(spec['wav'].shape[0])
			for i in range(dw.size):
				dw[i] = np.abs(w_line - np.nanmedian(spec['wav'][i,:]))
			order = dw == np.min(dw)
			w = spec['wav'][order,:]
			f = spec['flux'][order,:]
		else:
			w = spec['wav']
			f = spec['flux']

	f = f[~np.isnan(w)]
	w = w[~np.isnan(w)]

	w = w[~np.isnan(f)]
	f = f[~np.isnan(f)]

	if cn == True:
		f = saph.utils.cont_norm(w,f,lower=0.5,upper=0.5)

	w_cor = w/(1-((bvcorr-gamma)/(c)))

	v = ((w_cor-w_line)/w_line)*c

	v_out = v[(v>-v_width)&(v<v_width)]
	f_out = f[(v>-v_width)&(v<v_width)]

	return v_out,f_out


def vac2air(w_vac):
	'''
	Vacuum to air conversion formula from Donald Morton 
	(2000, ApJ. Suppl., 130, 403) is used for the refraction index, 
	which is also the IAU standard:
	http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

	Parameters
	----------
	w_vac : array-like
		Array of vacuum wavelengths assumed to be in Angstroms

	Returns
	-------
	w_air : array-like
		Array of air wavelengths converted from w_vac
	'''

	s = 10**4/w_vac

	n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)

	w_air = w_vac/n

	return w_air



