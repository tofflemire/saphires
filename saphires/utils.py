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

Description:
A collection of utility functions used in the SAPHIRES package.
'''

# ---- Standard Library
import sys
import copy as copy
# ----

# ---- Third Party
import numpy as np
from scipy import interpolate
# ---- 

# ---- Project
from saphires.extras import bspline_acr as bspl
# ----

def cont_norm(w,f,w_width=200.0):
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

	Returns
	-------
	f_norm : array-like
		Continuum normalized flux array

	'''
	norm_space = w_width/(w[1]-w[0])

	x = np.arange(f.size,dtype=float)
	spl = bspl.iterfit(x, f, maxiter = 15, lower = 0.3, 
	                   upper = 2.0, bkspace = norm_space, 
	      	           nord = 3 )[0]
	cont = spl.value(x)[0]
	f_norm = f/cont

	return f_norm


def spec_trim(w_tar,f,w_range,temp_trim,trim_style='clip'):
	'''
	A function to select certain regions of a spectrum with which
	to compute the broadedning function. 

	trim_style - refers to how you want to deal with the bad regions
	- 'clip' - remove data all together. This creates edges that can cause noise in the BF
	- 'lin' - linearly interpolates over clipped regions
	- 'spl' - interpolated over the clipped regions with a cubic spline - don't use this option.
	

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


def bf_prepare(t_f_names,t_spectra,temp_name,temp_spec,m,oversample=1,
               quiet=False,trap_apod=0,cr_trim=-0.5,trim_style='clip'):
	'''
	A function to prepare a target spectral dictionary (this can be a 
	dictionary for a single spectrum or multiple spectra) with a template 
	spectral dictionary (this must be a single spectrum) in order to 
	compute the broadedning function.
	
	oversample - 
	A key parameter of this function is "oversample". It sets the logarithmic 
	spacing of the wavelength resampling and the corresponding velocity 
	'resolution' of the broadening function. Generally, a higher oversampling
	will produce better results, but it very quickly becomes expensive for two
	reasons. One, your arrays become longer, and two, you have to increase the m
	value (a proxy for the velocity regime probed) to cover the same velocity range. 

	A low oversample value is problematic because if the intrinsitic width of your
	tempate and target lines are the same, the broadening function should be a delta
	function. The height/width of this function will depened on the location of 
	velocity grid points in the BF in an undersampled case. This makes measurements 
	of the RV and especially flux ratios problematic. 

	If you have a very narrow lined template and a broad target spectrum, an 
	oversample = 3 should be adequeate. In practice though, you may want something 
	around 5. If you're unsure what value to use, look at the BF with low smoothing
	(i.e. high R). If the curves are jagged and look undersampled, increase the 
	oversample parameter.

	The target spectral dictionary must have this format of sub-arrays
	assocated with each keyword:
	
	[0] - native flux array (inverted)
	[1] - native wavelength array
	[2] - native wavelength spacing value
	[3] - Order Central Wavelength
	[4] - 0 (single ints; this function will this with an array)
	[5] - 0 (single ints; this function will this with an array)
	[6] - 0 (single ints; this function will this with an array)
	[7] - 0 (single ints; this function will this with an array)
	[8] - wavelength region
	[9] - 0 (single int; this function will input the template name)
	[10]- template wavelength region
	[11]- 0 (single int; this function will this with an array)

	The template spectral dictionary must have this format of sub-arrays
	associated with its single keyword:
	
	[0] - native flux array (inverted)
	[1] - native wavelength array
	[2] - native wavelength spacing value
	[3] - iOrder Central Wavelength
	[4] - wavelength region
	
	The output will be an updated version of the input target spectral 
	dictionary. It will have the following format:

	[0] - native flux array (inverted)
    [1] - native wavelength array
    [2] - native wavelength spacing value
    [3] - Order Central Wavelength
    [4] - resampled flux array (inverted)
    [5] - resampled wavelength array
    [6] - resampled template flux array (inverted)
    [7] - velocity array - to be plot against the BF
    [8] - wavelength region
    [9] - template name
    [10]- template wavelength region
    [11]- broadening funciton

	Parameters
	----------
	t_f_names: array-like

	t_spectra : python dictionary

	temp_name : array-like
	
	temp_spec : python dictionary


	m : int

	oversample : float
		Default is 1.

    quiet : bool
    	False

    trap_apod : float
    	Detault is 0, i.e. no apodization.

    cr_trim	:
    	-0.5

    trim_style : 


	Returns
	-------

	'''
	#########################################
	#This part "prepares" the spectra

	spectra = copy.deepcopy(t_spectra)

	for i in range(t_f_names.size):
		#print t_f_names[i]
		
		spectra[t_f_names[i]][9] = temp_name
		
		w_range = spectra[t_f_names[i]][8]
		w_range_iter = spectra[t_f_names[i]][16]

		w_tar = spectra[t_f_names[i]][1]
		flux_tar = spectra[t_f_names[i]][0]
		
		temp_trim = temp_spec[temp_name][4]
		w_temp = temp_spec[temp_name][1]
		flux_temp = temp_spec[temp_name][0]
		
		#This gets rid of large emission lines and CRs by interpolating over them.
		if np.min(flux_tar) < cr_trim:
			f_tar = interpolate.interp1d(w_tar[flux_tar > cr_trim],flux_tar[flux_tar > cr_trim])
			w_tar = w_tar[(w_tar >= np.min(w_tar[flux_tar > cr_trim]))&(w_tar <= np.max(w_tar[flux_tar > cr_trim]))]
			flux_tar = f_tar(w_tar)

		if np.min(flux_temp) < cr_trim:
			f_temp = interpolate.interp1d(w_temp[flux_temp > cr_trim],flux_temp[flux_temp > cr_trim])
			w_temp = w_temp[(w_temp >= np.min(w_temp[flux_temp > cr_trim]))&(w_temp <= np.max(w_temp[flux_temp > cr_trim]))]
			flux_temp = f_temp(w_temp)

		w_tar,flux_tar = bf_spec_trim(w_tar,flux_tar,w_range,temp_trim,trim_style=trim_style)
		w_tar,flux_tar = bf_spec_trim(w_tar,flux_tar,w_range_iter,'*',trim_style=trim_style)

		if w_tar.size == 0:
			if quiet==False:
				print t_f_names[i],w_range
				print "No overlap between target and template."
				print ' '
			spectra[t_f_names[i]][5] = 0.0
			spectra[t_f_names[i]][15] = 0
			continue

		f_tar = interpolate.interp1d(w_tar,flux_tar)
		f_temp = interpolate.interp1d(w_temp,flux_temp)

		min_w = np.max([np.min(w_tar),np.min(w_temp)])

		max_w = np.min([np.max(w_tar),np.max(w_temp)])

		#Using the wavelength spacing for the template which is more
		#technically motivated
		min_dw=np.min([temp_spec[temp_name][2],spectra[t_f_names[i]][2]])

		#inverse of the spectral resolution
		r = min_dw/max_w/oversample #Over sampled by roughly a factor of 3 (assuming the resolution element is 3 pixels)

		#velocity spacing in km/s
		stepV=r*2.997924*10**5

		#velocity array
		vel=stepV*(np.arange(m)-m/2)
		
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

		w1t,t_rflux = bf_spec_trim(w1t,t_rflux,w_range,temp_trim,trim_style=trim_style)
		w1t,t_rflux = bf_spec_trim(w1t,t_rflux,w_range_iter,temp_trim,trim_style=trim_style)

		w1t_temp,temp_rflux = bf_spec_trim(w1t_temp,temp_rflux,w_range,temp_trim,trim_style=trim_style)
		w1t_temp,temp_rflux = bf_spec_trim(w1t_temp,temp_rflux,w_range_iter,temp_trim,trim_style=trim_style)

		if (w1t.size/2.0 % 1) != 0:
			w1t=w1t[0:-1]
			t_rflux = t_rflux[0:-1]
			temp_rflux = temp_rflux[0:-1]


		##resampled fluxes and flux normalization if need be
		#if tar_norm == False:
		#	t_rflux = f_tar(w1t)
		#
		#if tar_norm == True:
		#	flux = 1.0-spectra[t_f_names[i]][0]
		#	x = np.arange(flux.size,dtype=float)
		#	spl = bspl.iterfit(x, flux, maxiter = 15, 
		#	                   lower = 0.3, upper = 2.0, bkspace = 40000, 
		#	                   nord = 3)[0]
		#	cont = spl.value(x)[0]
		#
		#	t_nflux = 1.0-(flux/cont)
		#
		#	f_t = interpolate.interp1d(spectra[t_f_names[i]][1],
		#	                           t_nflux)
		#
		#	t_rflux = f_t(w1t)
		#
		#if temp_norm == False:
		#	temp_rflux = f_temp(w1t)
		#
		#if temp_norm == True:
		#	flux = 1.0 - f_temp(w1t)
		#
		#	x = np.arange(flux.size,dtype=float)
		#	spl = bspl.iterfit(x, flux, maxiter = 15, 
		#	                   lower = 0.3, upper = 2.0, bkspace = 40000, 
		#	                   nord = 3 )[0]
		#	cont = spl.value(x)[0]
		#
		#	temp_rflux = 1.0-(flux/cont)

		#if (w1t.size/2.0 % 1) != 0:
		#	w1t=w1t[0:-1]

		#if (temp_rflux.size/2.0 % 1) != 0:
		#	temp_rflux=temp_rflux[0:-1]

		#if (t_rflux.size/2.0 % 1) != 0:
		#	t_rflux=t_rflux[0:-1]

		if trap_apod > 0:
			trap_apod_fun = np.ones(w1t.size)
			slope = 1.0/np.int(w1t.size*trap_apod)
			y_int = slope*w1t.size
			trap_apod_fun[:np.int(w1t.size*trap_apod)] = slope*np.arange(np.int(w1t.size*trap_apod),dtype=float)
			trap_apod_fun[-np.int(w1t.size*trap_apod)-1:] = -slope*(np.arange(np.int(w1t.size*trap_apod+1),dtype=float)+(w1t.size*(1-trap_apod))) + y_int

			temp_rflux = temp_rflux * trap_apod_fun
			t_rflux = t_rflux * trap_apod_fun

		if w1t.size < m:
			if quiet == False:
				print t_f_names[i],t_spectra[t_f_names[i]][8]
				print "The target mask region is smaller for the m value."
				print w1t.size,'versus',m
				print "You can either reduce m or remove this order from the input or don't worry about it."
				print ' '
			spectra[t_f_names[i]][5] = 0.0	
			spectra[t_f_names[i]][15] = 0
			continue

		spectra[t_f_names[i]][4] = t_rflux
		spectra[t_f_names[i]][5] = w1t
		spectra[t_f_names[i]][6] = temp_rflux
		spectra[t_f_names[i]][7] = vel
		spectra[t_f_names[i]][9] = temp_name
		spectra[t_f_names[i]][10] = temp_spec[temp_name][4]

	return spectra







