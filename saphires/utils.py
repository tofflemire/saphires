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
# ----

# ---- Third Party
import numpy as np
from scipy import interpolate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle as pkl
from matplotlib.backends.backend_pdf import PdfPages
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


def apply_shift(t_f_names,t_spectra,rv_shift):
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

	Returns
	-------
	spectra_out : python dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will be a copy of t_specrta, but with updates to the following keywords.

		['nwave']    - The shifted wavelength array
		['nflux']    - The shifted flux array
		['rv_shift'] - The value the spectrum was shifted in km/s

	'''

	spectra_out = copy.deepcopy(t_spectra)

	for i in range(t_f_names.size):
		w_unshifted = spectra_out[t_f_names[i]]['nwave']
	
		w_shifted=w_unshifted/(1-(-rv_shift/(2.997924*10**5)))
		
		f_shifted_f=interpolate.interp1d(w_shifted,spectra_out[t_f_names[i]]['nflux'])
	
		shift_trim = ((w_unshifted>=np.min(w_shifted))&(w_unshifted<=np.max(w_shifted)))
	
		w_unshifted = w_unshifted[shift_trim]
	
		spectra_out[t_f_names[i]]['nwave'] = w_unshifted
	
		f_out=f_shifted_f(w_unshifted)
	
		spectra_out[t_f_names[i]]['nflux'] = f_out

		w_range = spectra_out[t_f_names[i]]['w_region']

		if w_range != '*':
			w_split = np.empty(0)
			w_rc1 = w_range.split('-')
			for j in range(len(w_rc1)):
				for k in range(len(w_rc1[j].split(','))):
					w_split = np.append(w_split,np.float(w_rc1[j].split(',')[k]))

			w_split_shift = w_split/(1-(-rv_shift/(2.997924*10**5)))

			w_range_shift = ''
			for j in range(w_split_shift.size):
				if (j/2.0 % 1) == 0: #even
					w_range_shift = w_range_shift + np.str(np.round(w_split_shift[j],2))+'-'
				if (j/2.0 % 1) != 0: #odd
					w_range_shift = w_range_shift + np.str(np.round(w_split_shift[j],2))+','

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

	pp=PdfPages(t_f_names[0].split('.')[0]+'_allplots.pdf')

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
					label='Amp1: '+np.str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
			ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		                                         gs_fit[3],gs_fit[4],
		                                         gs_fit[5],gs_fit[9]),
					lw=2,ls='--',color='r',
					label='Amp2: '+np.str(np.round(gs_fit[3]*gs_fit[5]*np.sqrt(2.0*np.pi),3)))
			ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		                                         gs_fit[6],gs_fit[7],
		                                         gs_fit[8],gs_fit[9]),
					lw=2,ls='--',color='g',
					label='Amp3: '+np.str(np.round(gs_fit[6]*gs_fit[8]*np.sqrt(2.0*np.pi),3)))
			ax.legend()
			
		if gs_fit.size == 7:
			#if func == gauss_rot_off:
			#	ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    #	                                     gs_fit[0],gs_fit[1],
		    #	                                     gs_fit[2],gs_fit[6]),
			#			lw=2,ls='--',color='b',
			#			label='Amp1: '+np.str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
			#
			#	ax.plot(vel[f_trim:-f_trim],rot_pro(vel[f_trim:-f_trim],
		    #	                                     gs_fit[3],gs_fit[4],
		    #	                                     gs_fit[5],gs_fit[6]),
			#			lw=2,ls='--',color='r',
			#			label='Amp2: '+np.str(np.round(gs_fit[3],3)))

			if func == d_gaussian_off:
				ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    	                                     gs_fit[0],gs_fit[1],
		    	                                     gs_fit[2],gs_fit[6]),
						lw=2,ls='--',color='b',
						label='Amp1: '+np.str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2.0*np.pi),3)))
				
				ax.plot(vel[f_trim:-f_trim],gaussian_off(vel[f_trim:-f_trim],
		    	                                     gs_fit[3],gs_fit[4],
		    	                                     gs_fit[5],gs_fit[6]),
						lw=2,ls='--',color='r',
						label='Amp2: '+np.str(np.round(gs_fit[3]*gs_fit[5]*np.sqrt(2.0*np.pi),3)))
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
		f.write(np.str(np.round(gs_fit[peak_ind[0]-1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[0]],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[0]+1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[0]-1]*
		                        gs_fit[peak_ind[0]+1]*np.sqrt(2*np.pi),2))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[1]-1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[1]],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[1]+1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[1]-1]*
		                        gs_fit[peak_ind[1]+1]*np.sqrt(2*np.pi),2))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[2]-1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[2]],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[2]+1],4))+'\t')
		f.write(np.str(np.round(gs_fit[peak_ind[2]-1]*
		                        gs_fit[peak_ind[2]+1]*np.sqrt(2*np.pi),2))+'\t')

	if gs_fit.size==7:
		if fit_int[0]>fit_int[1]:
			f.write(np.str(np.round(gs_fit[0],4))+'\t')
			f.write(np.str(np.round(gs_fit[1],4))+'\t')
			f.write(np.str(np.round(gs_fit[2],4))+'\t')
			f.write(np.str(np.round(fit_int[0],2))+'\t')
			f.write(np.str(np.round(gs_fit[3],4))+'\t')
			f.write(np.str(np.round(gs_fit[4],4))+'\t')
			f.write(np.str(np.round(gs_fit[5],4))+'\t')
			f.write(np.str(np.round(fit_int[1],2))+'\t')
		else:
			f.write(np.str(np.round(gs_fit[3],4))+'\t')
			f.write(np.str(np.round(gs_fit[4],4))+'\t')
			f.write(np.str(np.round(gs_fit[5],4))+'\t')
			f.write(np.str(np.round(fit_int[1],2))+'\t')
			f.write(np.str(np.round(gs_fit[0],4))+'\t')
			f.write(np.str(np.round(gs_fit[1],4))+'\t')
			f.write(np.str(np.round(gs_fit[2],4))+'\t')
			f.write(np.str(np.round(fit_int[0],2))+'\t')

	if gs_fit.size==4:
		f.write(np.str(np.round(gs_fit[0],4))+'\t')
		f.write(np.str(np.round(gs_fit[1],4))+'\t')
		f.write(np.str(np.round(gs_fit[2],4))+'\t')
		f.write(np.str(np.round(gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi),2))+'\t')
	
	f.write(np.str(np.round(rchis,3))+'\t')
	f.write(np.str(np.round(rv_weight,3))+'\t')

	f.write(target+'\t')
	f.write(template+'\n')

	f.close()

	return


def cont_norm(w,f,w_width=200.0,maxiter=15,lower=0.3,upper=2.0,nord=3):
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
	norm_space = w_width/(w[1]-w[0])

	x = np.arange(f.size,dtype=float)
	spl = bspl.iterfit(x, f, maxiter = maxiter, lower = lower, 
	                   upper = upper, bkspace = norm_space, 
	      	           nord = nord )[0]
	cont = spl.value(x)[0]
	f_norm = f/cont

	return f_norm


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
	FWHM = (2.997924*10**5)/R
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


def prepare(t_f_names,t_spectra,temp_spec,oversample=1,
    quiet=False,trap_apod=0,cr_trim=-0.1,trim_style='clip',
    vel_spacing='orders'):
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
		This parameter sets the value below which emission features are removed. 
		Emission is this case is negative becuase the spectra are inverted. The
		value must be negative. Points below this value are linearly interpolated
		over. The defulat value is -0.1. If you don't want to clip anything, set 
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
		
		min_w_orders[i] = np.max([np.min(w_tar),np.min(w_temp)])
		max_w_orders[i] = np.min([np.max(w_tar),np.max(w_temp)])
	
		min_dw_orders[i]=np.min([temp_spec['ndw'],spectra[t_f_names[i]]['ndw']])
	
	min_dw = np.min(min_dw_orders)
	min_w = np.min(min_w_orders)
	max_w = np.max(max_w_orders)

	if vel_spacing == 'uniform':
		r = np.min(min_dw/max_w/oversample)
		#velocity spacing in km/s
		stepV=r*2.997924*10**5

	if type(vel_spacing) == float:
		stepV = vel_spacing
		r = stepV / (2.997924*10**5)
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
		if np.min(flux_tar) < cr_trim:
			f_tar = interpolate.interp1d(w_tar[flux_tar > cr_trim],flux_tar[flux_tar > cr_trim])
			w_tar = w_tar[(w_tar >= np.min(w_tar[flux_tar > cr_trim]))&
						  (w_tar <= np.max(w_tar[flux_tar > cr_trim]))]
			flux_tar = f_tar(w_tar)

		if np.min(flux_temp) < cr_trim:
			f_temp = interpolate.interp1d(w_temp[flux_temp > cr_trim],flux_temp[flux_temp > cr_trim])
			w_temp = w_temp[(w_temp >= np.min(w_temp[flux_temp > cr_trim]))&
							(w_temp <= np.max(w_temp[flux_temp > cr_trim]))]
			flux_temp = f_temp(w_temp)

		w_tar,flux_tar = spec_trim(w_tar,flux_tar,w_range,temp_trim,trim_style=trim_style)

		if w_tar.size == 0:
			if quiet==False:
				print(t_f_names[i],w_range)
				print("No overlap between target and template.")
				print(' ')
			spectra[t_f_names[i]]['vwave'] = 0.0
			spectra[t_f_names[i]]['order_flag'] = 0
			continue

		f_tar = interpolate.interp1d(w_tar,flux_tar)
		f_temp = interpolate.interp1d(w_temp,flux_temp)

		min_w = np.max([np.min(w_tar),np.min(w_temp)])

		max_w = np.min([np.max(w_tar),np.max(w_temp)])


		if vel_spacing == 'orders':
			#Using the wavelength spacing of the most densely sampled spectrum
			min_dw=np.min([temp_spec['ndw'],spectra[t_f_names[i]]['ndw']])
	
			#inverse of the spectral resolution
			r = min_dw/max_w/oversample 
	
			#velocity spacing in km/s
			stepV = r * 2.997924*10**5
		
		#the largest array length between target and spectrum
		#conditional below makes sure it is even
		max_size = np.int(np.log(max_w/(min_w+1))/np.log(1+r))
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
			slope = 1.0/np.int(w1t.size*trap_apod)
			y_int = slope*w1t.size
			trap_apod_fun[:np.int(w1t.size*trap_apod)] = slope*np.arange(np.int(w1t.size*trap_apod),dtype=float)
			trap_apod_fun[-np.int(w1t.size*trap_apod)-1:] = -slope*(np.arange(np.int(w1t.size*trap_apod+1),dtype=float)+(w1t.size*(1-trap_apod))) + y_int

			temp_rflux = temp_rflux * trap_apod_fun
			t_rflux = t_rflux * trap_apod_fun

		spectra[t_f_names[i]]['vflux'] = t_rflux
		spectra[t_f_names[i]]['vwave'] = w1t
		spectra[t_f_names[i]]['vflux_temp'] = temp_rflux
		spectra[t_f_names[i]]['vel_spacing'] = stepV
		spectra[t_f_names[i]]['w_region_temp'] = temp_spec['w_region']

	return spectra


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


def region_select_pkl(target,template=None,tar_stretch=True,
    temp_stretch=True,reverse=False,dk_wav='wav',dk_flux='flux'):
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

	Returns
	-------
	None

	'''
	l_range = []
	
	def press_key(event):
		if event.key == 'b':
			l_range.append(np.int(np.round(event.xdata,2)))

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

	#----- Reading in and Formatiing ---------------	
	if template == None:
		template = copy.deepcop(target)

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
	#-------------------------------------------------

	plt.ion()

	i = 0
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
		
		fig,ax=plt.subplots(2,sharex=True)

		ax[0].set_title('Target - '+np.str(i_ind))
		ax[0].plot(w,flux)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[0].axvline(l_range[j],ls=':',color='red')
		ax[0].set_ylabel('Flux')
		if tar_stretch == True:
			ax[0].axis([np.min(w),np.max(w),
		    	       np.median(flux)-np.median(flux)*0.5,
		        	   np.median(flux)+np.median(flux)*0.5])
		ax[0].grid(b=True,which='both',axis='both')

		ax[1].set_title('Template - '+np.str(i_ind))
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
		ax[1].grid(b=True,which='both',axis='both')

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

		fig.canvas.mpl_disconnect(cid)

		plt.cla()

		plt.close()
        
	return


def region_select_vars(w,f,tar_stretch=True,reverse=False):
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

	#----- Reading in and Formatiing ---------------	
	if (w.ndim == 1):
		order = 1

	if (w.ndim > 1):
		order=w.shape[0]
	#-------------------------------------------------

	plt.ion()

	i = 0
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

		fig,ax=plt.subplots(2,sharex=True)

		ax[0].set_title('Target - '+np.str(i_ind))
		ax[0].plot(w_plot,flux_plot)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[0].axvline(l_range[j],ls=':',color='red')
		ax[0].set_ylabel('Flux')
		if tar_stretch == True:
			ax[0].axis([np.min(w_plot),np.max(w_plot),
		    	       np.median(flux_plot)-np.median(flux_plot)*0.5,
		        	   np.median(flux_plot)+np.median(flux_plot)*0.5])
		ax[0].grid(b=True,which='both',axis='both')

		ax[1].plot(w_plot,flux_plot)
		if len(l_range) > 0:
			for j in range(len(l_range)):
				ax[1].axvline(l_range[j],ls=':',color='red')
		ax[1].set_ylabel('Flux')
		ax[1].set_xlabel('Wavelength')		

		ax[1].grid(b=True,which='both',axis='both')

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
			
			t_ind[(w_tar >= np.float(w_start)) & (w_tar <= np.float(w_end))] = True

		if temp_trim != '*':

			n_regions=len(str(temp_trim).split('-'))

			for j in range(n_regions+1):
				if j == 0:
					w_start=str(temp_trim).split('-')[j]
					if np.min(w_tar) < np.float(w_start):
						t_ind[w_tar <= np.float(w_start)] = False

				if (j > 0) & (j < n_regions-1):
					i_range=str(temp_trim).split('-')[j]
					w_start=i_range.split(',')[0]
					w_end=i_range.split(',')[1]
					if (np.min(w_tar) > np.float(w_start)) & (np.min(w_tar) < np.float(w_end)):
						t_ind[w_tar < np.float(w_end)] = False

					if (np.min(w_tar) < np.float(w_start)) & (np.max(w_tar) > np.float(w_end)):
						t_ind[(w_tar > np.float(w_start)) & 
								(w_tar < np.float(w_end))] = False

					if (np.max(w_tar) > np.float(w_start)) & (np.max(w_tar) < np.float(w_end)):
						t_ind[w_tar > np.float(w_start)] = False

				if j == n_regions:
					w_end=str(temp_trim).split('-')[-1]
					if (np.max(w_tar) < np.float(w_end)):
						t_ind[w_tar > np.float(w_end)] = False
					if (np.min(w_tar) > np.float(w_end)):
						t_ind[w_tar > np.float(w_end)] = False

	if ((temp_trim != '*') & (w_range == '*')):

		t_ind = np.zeros(w_tar.size,dtype=bool)

		n_regions = len(str(temp_trim).split(','))

		for j in range(n_regions):
			w_start=str(temp_trim).split(',')[j].split('-')[0]
			w_end=str(temp_trim).split(',')[j].split('-')[1]
			t_ind[(w_tar >= np.float(w_start)) & 
				(w_tar <= np.float(w_end))] = True

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
			          np.float(f_s_ccf[:ccf_i].size))
		if ccf_i >=0:
			ccf[i] = (np.sum( f_s_ccf[ccf_i:] * 
			                 np.roll(f_t_ccf,ccf_i)[ccf_i:] ) / 
					  np.float(f_s_ccf[ccf_i:].size))

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
	xo = np.float(xo)
	yo = np.float(yo)    
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

	z = g.ravel()

	return z


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


# THIS IS IMPORTANT CODE THAT NEEDS TO BE PUT AT THE TOP OF THE bf.compute AND
# xc.todcor FUNCTIONS
#if w1t.size < m:
#	if quiet == False:
#		print t_f_names[i],t_spectra[t_f_names[i]][8]
#		print "The target mask region is smaller for the m value."
#		print w1t.size,'versus',m
#		print "You can either reduce m or remove this order from the input or don't worry about it."
#		print ' '
#	spectra[t_f_names[i]][5] = 0.0	
#	spectra[t_f_names[i]][15] = 0
#	continue

#velocity array
#vel=stepV*(np.arange(m)-m//2)

