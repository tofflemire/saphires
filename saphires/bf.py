'''
############################ SAPHIRES bf ##############################
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
A collection of SAPHIRES functions that compute spectral line 
broadening functions (BFs) and analyze their results. 
'''

# ---- Standard Library
import copy as copy
# ----

# ---- Third Party
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle as pkl
# ---- 

# ---- Project
from saphires import utils
# ---- 

def compute(t_f_names,t_spectra,vel_width=200,quiet=False):
	'''
	Compute the spectral lines broadening function (BF) between a target and
	template spectrum. 

	The input target spectrum dictionary should have been "prepared" with 
	saphires.utils.prepare with the corresponding template before being 
	input to this function.

	The output BF will need to be smoothed for you to do anything useful 
	with it. This is done automatically in saphire.bf.analysis, but if you 
	prefer to roll your own, heads up. 

	When selecting the velocity width, play around with different values. 
	You don't want to compute the BF over a velocity range larger than you
	need to for two reasons, 1) this function runs slowly, and 2) noisey 
	spectra can put extra "power" into noisey side-bands regions that can 
	distort the shape and amplitude of your line profile(s). 
		- You probably won't know the right range ahead of time so start 
		  broad
		- Think about applying a shift to your spectrum with 
		  saphires.utils.apply_shit to center the feature(s) you care about, 
		  allowing you to compute the BF over a smaller velocity range
		- It is good to have an equal amount of velocity space computed 
		  within the features you care about and outside of them

	Parameters
    ----------
    t_f_names: array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum that has been prepared with 
		the utils.prepare function with a template spectrum. 

	vel_width : float
		The range over which to compute the two-dimensional cross correlation.
		Larger values take longer to run. Small values could exclude a peak 
		you care about. The default value is 200 km/s. 

	quiet : bool
    	Specifies whether messaged are printed to the teminal. Specifically, if 
    	there is not enough velocity space in the input spectra,this function 
    	will print and error. The default value is False.

	Returns
    -------
	spectra : python dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will have 2 new keywords as a result of this function. And is a copy of 
		t_spec1.

		['vel'] - Velocity array over which the BF is computed
		['bf'] - The unsmoothed BF

		It also updates the values for the following keyword under the right 
		conditions:

		['order_flag'] 	- order flag will be updated to 0 if the order has less 
						  velocity space than is asked to compute.
	'''
	#########################################

	spectra = copy.deepcopy(t_spectra)

	#The following does a bit of logic to see it the function can save time
	#by using the same design matrix for mutiple science spectra
	w_unique = np.zeros(t_f_names.size,dtype='S200')

	for i in range(t_f_names.size):
		if type(spectra[t_f_names[i]]['nwave']) != float:

			w_unique[i] = (np.str(np.min(spectra[t_f_names[i]]['nwave']))+' '+
		    	           np.str(np.max(spectra[t_f_names[i]]['nwave']))+' '+
		        	       np.str((spectra[t_f_names[i]]['nwave']).size))
		if type(spectra[t_f_names[i]]['nwave']) == float:
			w_unique[i] = '0'

	w_unique_ind = np.unique(w_unique[w_unique!='0'])

	#Run through each spectrum requiring unique design matrix to compute the BF
	for i in range(w_unique_ind.size):
		inds = np.where(w_unique == w_unique_ind[i])[0]
		
		m = np.int(vel_width / spectra[t_f_names[inds[0]]]['vel_spacing'])
		if (m/2.0 % 1) == 0:
			m=m-1

		des=utils.bf_map(spectra[t_f_names[inds[0]]]['vflux_temp'],m)
		
		u,ww,vt=np.linalg.svd(des, full_matrices=False)

		if ww.size < m:
			if quiet==False:
				print(t_f_names[inds[0]],t_spectra[t_f_names[inds[0]]]['w_region'])
				print("The target region is too small for the vel_width value.")
				print(ww.size*spectra[t_f_names[inds[0]]]['vel_spacing'],' versus ', vel_width)
				print("You can either reduce vel_width or remove this order from the input or don't worry about it.")
				print(' ')
			spectra[t_f_names[inds[0]]]['order_flag'] = 0
			continue

		for j in range(inds.size):
			bf_sols,sig = utils.bf_solve(des,u,ww,vt,spectra[t_f_names[inds[j]]]['vflux'],m)

			vel = spectra[t_f_names[inds[j]]]['vel_spacing']*(np.arange(m)-m//2)

			spectra[t_f_names[inds[j]]]['bf']=bf_sols[m-1]
			spectra[t_f_names[inds[j]]]['vel']=vel

	return spectra


def analysis(t_f_names,t_spectra,sb='sb1',fit_trim=20,
    text_out=False,text_name=False,single_plot=False,
    p_rv=False,prof='g',R=50000.0,R_ip=50000.0,e=0.75):
	'''
	A function to analyze broadening functions. This will smooth the 
	BF and attempt to fit a specified number of line profiles to it
	in order to measure stellar RVs and determine their flux ratios,
	in the case of a spectrscopic binary or triple. 

	DISCLAIMER - This function is a beast, but it does most of what 
	I want it to do with the values below with the default values 
	that are hard coded below. Eventually, there should be keyword 
	arguments for the fit bounds etc., but for now if you need to 
	change them, I'm afraid you'll have to alter this code.
	
	Parameters
	----------
	t_f_names: array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum. Output of one of the 
		saphires.io read-in functions.

	sb : str
		Specified the type of system: 'sb1' is single-lined, will fit one of the 
		profiles specified in prof, 'sb2' will fit two profiles, 'sb3' will fit 
		three. Only Gaussian profiles are currenly supported for the 'sb2' and 
		'sb3' options. Any more profiles and I'm sorry, you're going to have to 
		roll your own. The default is 'sb1'.

	fit_trim : int
		The amount of points to trim from the edges of the BF before the fit is 
		made. The edges are usually noisey. Units are in wavelength spacings. 
		The default is 20.

	text_out : bool
		Whether to write the results of the fit (fit paramters, including RVs; profile 
		integrals to compute flux ratios) to a text file. It has a very descriptive 
		header. The make of the file is specified with the text_name keyword. Note that 
		running this functin multiple time will append results to this file rather than 
		overwriting them. This is useful when comparing different templates. The 
		default is False.

	text_name : bool; str
		If text_out is set to True, this value will be set as the file's name. The 
		function does not append a suffix so include your own .dat, .txt, etc.
		If text_out is set to True and text_name is left as False, the file will be 
		named, 'bf_text.dat'. The default value is False.

	single_plot : bool 
		If true, a multipage PDF is output with the results of the analysis. Two pages 
		are output per order, the first is a two panel figure with the target and 
		template spectrum in their continuum normlaized, inverted form. The second is 
		the smoothed BF with the fit overplotted. The file will have the name of the 
		input file, where the suffix (e.g., .p or .fits) is replaced with 
		'_allplots.pdf'. The default is False.

    p_rv : bool; array-like
    	The initial guess for profile fitting. Necessary for complex or noisey BFs. 
    	When used it is a single value for sb = 'sb1', an array of two values for 
    	sb = 'sb2', and so on. The default is False.

    prof : str 
    	The type of profile to fit. Options are 'g', for Gaussian, 'r', a rotatinoally 
    	broadened profile convolved with the smoothing profile (the 
    	smoothing/intrumental profile is specified with the R parameter). 
		- The Gaussian profile has 4 parameters: amplitude (not normalized), center 
		  (this is the RV), standard deviation (sigma), and vertical offset. 
		- The rotataionally broadened profile has 4 paramteres: amplitude (normalized, 
		  i.e., the amplitude is the profile integral), center (RV), width (vsini), and 
		  vertical offset. The profile uses a linear limb-darkening law that is 
		  specified with the 'e' keyword (limb darkening is not fit). Analytic formula 
		  from Gray 2005.
    	The default is 'g', a Gaussian.

    R : float
    	Defines the smoothing element. The first staring place should the be 
    	spectrograph's resolution. Smoothing at higher resolutions doesn't make sense.
    	Additional smoothing, i.e., a lower resolution may be advantageous for noisey 
    	BFs or very heavily rotationally broadened profiles. The rotatinoally broadened 
    	profile is convolved with this level of smoothing as well. The default is 50000.

	e : float
		Linear limb darkening parameter. Default is 0.75, 
		appropriate for a low-mass star.

	Returns
	-------
	spectra : dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will have 2 new keywords as a result of this function.

		['bf_smooth'] - BF smoothed by the input spectral resolution
		['bf_fits']   - Profile fit parameters to the BF. For every profile fit there 
						will be 3 elements that are orders as follows: amplitude, center, 
						width, and then a single  shared vertical offset.
						The details for what these parameters correspond to depend on the 
						profile being fit and are provided in the prof keyword description
						above. 
						Note that if you have shifted the spectra, these values correpsond
						to the fit to the shifted specrtum, they are unaware of any shifts 
						that have been made.
		['bf_rv']     - The rv values that have had any rv shift added back in. If you have 
						not shifted the spectra, these values will be the same as those 
						contained in the 'bf_fits' array. For sb2's or sb3's the rv are 
						ordered by the integral of their profile, which should correspond to
						the primary, seceondary, then tertiary. 

	'''
	spectra = copy.deepcopy(t_spectra)
	
	for_plotting={}
	t_f_ind=np.ones(t_f_names.size,dtype=bool)

	for i in range(t_f_names.size):

		if type(spectra[t_f_names[i]]['order_flag']) == 0:
			t_f_ind[i] = False
			continue

		vel = spectra[t_f_names[i]]['vel']

		ord_center = spectra[t_f_names[i]]['wav_cent']

		if R != 0:
			FWHM_lam = ord_center/R

			v_smooth_sig = (2.9979*10**5)*(FWHM_lam/(2.0*np.sqrt(2.0*np.log(2.0))))/ord_center

			bf_smooth = np.convolve(spectra[t_f_names[i]]['bf'],
	        	                	utils.gaussian_off(vel,
	        	                	                   (vel[1]-vel[0])/np.sqrt(2.0*np.pi*v_smooth_sig**2),
	        	                	                   0,v_smooth_sig,0),mode='same')
		else:
			bf_smooth = spectra[t_f_names[i]]['bf']

		for_plotting[t_f_names[i]]=bf_smooth

		rv_low = np.min(vel[fit_trim+10:-(fit_trim+10)])
		rv_high = np.max(vel[fit_trim+10:-(fit_trim+10)])

		if sb == "sb1":
			if prof == 'g':
				func=utils.gaussian_off
	
				bounds = ([0.0001,rv_low,0.05,-1],[np.inf,rv_high,100,1])

				if p_rv == False:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds)
				else:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds,p0=[0.1,p_rv,1,0])

				fit_int = gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi)

			if prof == 'r':
				func = utils.make_rot_pro_ip(R,e)

				bounds = ([0.0005,rv_low,0.05,-1],[np.inf,rv_high,200,1])

				if p_rv == False:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds)
				else:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds,p0=[0.1,p_rv,20,0])

				fit_int = gs_fit[0]


			rchis=utils.RChiS(vel[fit_trim:-fit_trim],
			            	  bf_smooth[fit_trim:-fit_trim],
			            	  np.zeros(bf_smooth[fit_trim:-fit_trim].size)+
			            	  np.std(bf_smooth[fit_trim:-fit_trim]),
			            	  func,gs_fit)

			rv=gs_fit[1] #km/s


		if sb == "sb2":
			if prof == 'g':
				func = utils.d_gaussian_off

				bounds = ([0.01,rv_low,0.05,0.01,rv_low,0.05,-1],
				          [np.inf,rv_high,80,np.inf,rv_high,80,1])

				if p_rv == False:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds)
				else:
					gs_fit,gs_errors=curve_fit(func,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
					                           bounds=bounds,p0=[0.1,p_rv[0],1, 0.1,p_rv[1],1, 0])

				fit_int = np.array([gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi),
				                   gs_fit[3]*gs_fit[5]*np.sqrt(2*np.pi)])

				rv = np.array([gs_fit[1],gs_fit[4]])[np.argsort(fit_int)][::-1]
				#rv=gs_fit[np.where(gs_fit == np.max([gs_fit[0],gs_fit[3]]))[0][0]+1]

			#if prof == 'r':
			#	func=d_rot_pro
			#	if p_rv == False:
			#		gs_fit,gs_errors=curve_fit(func,\
      		#	                        spectra[t_f_names[i]][7][fit_trim:-fit_trim],
      		#	                        bf_smooth[fit_trim:-fit_trim],
      		#	                        bounds=([0.005,rv_low,0.05,
      		#	                                0.005,rv_low,0.05,
      		#	                                -1],
      		#	                                [1,rv_high,100,
      		#	                                1,rv_high,100,
      		#	                                1]))
			#	else:
			#		gs_fit,gs_errors=curve_fit(func,\
      		#	                        spectra[t_f_names[i]][7][fit_trim:-fit_trim],
      		#	                        bf_smooth[fit_trim:-fit_trim],
      		#	                        bounds=([0.005,rv_low,0.05,
      		#	                                0.005,rv_low,0.05,
      		#	                                -1],
      		#	                                [1,rv_high,100,
      		#	                                1,rv_high,100,
      		#	                                1]),
      		#	                        p0=[0.1,p_rv[0],1, 0.1,p_rv[1],1, 0])
			#
			#	fit_int = np.array([gs_fit[0],gs_fit[3]])

			#if prof == 'gr':
			#	func = gauss_rot_off
			#	if p_rv == False:
			#		gs_fit,gs_errors=curve_fit(func,\
      		#	                        spectra[t_f_names[i]][7][fit_trim:-fit_trim],
      		#	                        bf_smooth[fit_trim:-fit_trim],
      		#	                        bounds=([0.01,rv_low,0.05,
      		#	                                0.005,rv_low,0.05,
      		#	                                -1],
      		#	                                [np.inf,rv_high,15,
      		#	                                1,rv_high,200,
      		#	                                1]))
			#	else:
			#		gs_fit,gs_errors=curve_fit(func,\
      		#	                        spectra[t_f_names[i]][7][fit_trim:-fit_trim],
      		#	                        bf_smooth[fit_trim:-fit_trim],
      		#	                        bounds=([0.01,rv_low,0.05,
      		#	                                0.005,rv_low,3.0,
      		#	                                -1],
      		#	                                [1,rv_high,15,
      		#	                                1,rv_high,200,
      		#	                                1]),
      		#	                        p0=[0.1,p_rv[0],1, 0.1,p_rv[1],10, 0])
			#
			#	fit_int = np.array([gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi),gs_fit[3]])

			rchis=utils.RChiS(vel[fit_trim:-fit_trim],
			            	  bf_smooth[fit_trim:-fit_trim],
			            	  np.zeros(bf_smooth[fit_trim:-fit_trim].size)+
			            	  np.std(bf_smooth[fit_trim:-fit_trim]),
			            	  func,gs_fit)

			

		if sb == "sb3":
			func=t_gaussian_off

			bounds = ([0.01,rv_low,1,0.01,rv_low,1,0.01,rv_low,1,-1],
			          [np.inf,rv_high,15,np.inf,rv_high,15,np.inf,rv_high,15,1])

			if p_rv == False:
				guess = [0.1,-10,1, 0.1,0,1, 0.1,10,1, 0]
			else:
				guess = [0.1,p_rv[0],5, 0.1,p_rv[1],5, 0.1,p_rv[2],5, 0]

			gs_fit,gs_errors=curve_fit(t_gaussian_off,vel[fit_trim:-fit_trim],bf_smooth[fit_trim:-fit_trim],
			                           bounds=bounds,p0=guess)

			fit_int = np.array([gs_fit[0]*gs_fit[2]*np.sqrt(2*np.pi),
				                gs_fit[3]*gs_fit[5]*np.sqrt(2*np.pi),
				                gs_fit[6]*gs_fit[8]*np.sqrt(2*np.pi)])

			rv = np.array([gs_fit[1],gs_fit[4],gs_fit[7]])[np.argsort(fit_int)][::-1]

			rchis=untils.RChiS(vel[fit_trim:-fit_trim],
			            	   bf_smooth[fit_trim:-fit_trim],
			            	   np.zeros(bf_smooth[fit_trim:-fit_trim].size)+
			            	   np.std(bf_smooth[fit_trim:-fit_trim]),
			            	   func,gs_fit)

		for_plotting[t_f_names[i]]=[bf_smooth]
		for_plotting[t_f_names[i]].append(func)
		for_plotting[t_f_names[i]].append(gs_fit)

		spectra[t_f_names[i]]['bf_smooth'] = bf_smooth
		spectra[t_f_names[i]]['bf_fits'] = gs_fit
		spectra[t_f_names[i]]['bf_rv'] = rv+spectra[t_f_names[i]]['rv_shift']


		if text_out == True:
			gs_fit_out = copy.deepcopy(gs_fit)
			if sb == 'sb1': gs_fit_out[1] = gs_fit_out[1] + spectra[t_f_names[i]]['rv_shift']
			if sb == 'sb2': 
				gs_fit_out[1] = gs_fit_out[1] + spectra[t_f_names[i]]['rv_shift']
				gs_fit_out[4] = gs_fit_out[4] + spectra[t_f_names[i]]['rv_shift']
			if sb == 'sb3': 
				gs_fit_out[1] = gs_fit_out[1] + spectra[t_f_names[i]]['rv_shift']
				gs_fit_out[4] = gs_fit_out[4] + spectra[t_f_names[i]]['rv_shift']
				gs_fit_out[7] = gs_fit_out[7] + spectra[t_f_names[i]]['rv_shift']

			template_name=spectra[t_f_names[i]]['temp_name']

			if text_name == False:
				text_name_out = 'bf_text.dat'
			else:
				text_name_out = text_name

			if t_f_names[i] == 'Combined':
				utils.bf_text_output(text_name_out,t_f_names[i].split('[')[0]+'[Combined]',
				               		 spectra[t_f_names[i]]['temp_name'],gs_fit_out,rchis,rv_weight,fit_int)
			else:
				untils.bf_text_output(text_name_out,t_f_names[i],spectra[t_f_names[i]]['temp_name'],
				               		  gs_fit_out,rchis,rv_weight,fit_int)
	if single_plot == True:
		utils.bf_singleplot(t_f_names[t_f_ind],spectra,for_plotting,f_trim=fit_trim)

	return spectra


def save(t_f_names,t_spectra,outname,text_out=True):
	'''
	A function to save a SAPHIRES dictionary as a python pickle file.
	The pickle will be saved in whatever python version if currently 
	being used, i.e. 2 or 3. 

	Parameters
	----------
	t_f_names : array-like
	Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spectra : python dictionary
		SAPHIRES dictionary for the science spectrum that has been prepared with 
		the utils.prepare function with a template spectrum. 
	
	outname : str
		Name of the output pickle file. A '.p' suffix fill be attatched to the
		provided string. 

	text_out : bool
		Option to print a message to the terminal signifiying the output has
		been made. The defatul value is True.

	Returns
	-------
	None

	'''
	pkl.dump([t_f_names,t_spectra],open(outname+'.p','wb'))

	if text_out == True:
		print("Saved as: ", outname+'.p')

	return 

