'''
############################ SAPHIRES xc ##############################
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
A collection of SAPHIRES functions that perform Fourier cross 
correlcations in various flavors and analyze their results. 
'''

# ---- Standard Library
import sys
import copy as copy
# ----

# ---- Third Party
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# ---- 

# ---- Project
from saphires import utils
# ----

py_version = sys.version_info.major
if py_version == 3:
	p_input = input
if py_version == 2:
	p_input = raw_input
	
def todcor(t_f_names,t_spec1,t_spec2,vel_width=200.0,
    stamp_size=5,alpha_fit=True,guess=False,
    results=True,text_out=True):
	'''
	Two-Dimentional Cross Correlation
	Algorithm framework - Zucker & Mazeh 1994 (Appendix)
	http://adsabs.harvard.edu/abs/1994ApJ...420..806Z
	
	This function preforms a two-dimensional cross correlation on  
	two input SAPHIRES dictionaries that have been "prepared" by the 
	saphires.utils.prepare function.

	The t_spec1 and t_spec2 files should have already been run through 
	utils.prepare.

	The resultant 2 dimensional distribution is plotted as a contour
	plot with the primary RV on the y axis (the spectrum that is 
	prepared with the template that most closely matches the primary),
	and the secondary RV on the x axis. Here you can interactively 
	zoom aroun with the typical matplotlib funcationallity to find 
	the peak you want to fit. For a double-lined spectroscopic 
	binary there are going to be 4 peaks. Two will fall along the 
	unity line (dotted), which are not the ones you'll want to fit. 
	Two other peaks should appear on either side of the untity line, 
	either are possibly the one you want to fit.
	- If you have the alpha_fit parameter set to True, you will also
	  have a contour plot of the flux ratio, showing values less than
	  1.0. One of the suitable peaks will have a flux ratio below 1,
	  i.e. it will have a flux_ratio contour at the todcor peak location,
	  and one will not. 
	  As long as you haven't messed up the templates really bad, you'll
	  always want to pick the peak that has a flux ratio value < 1
	- If you do not fit the flux ratio value, choose the tallest peak. 
	  Sometimes this is difficult to tell, so I recommend setting 
	  alpha_fit to True. 
	You select the peak you want to fit by pressing 'm' when the cursor
	is over it.
	Pressing 'return' in the terminal will let the function continue.

	The procedure described above requires interactive capabilities 
	that may only work in an ipython session.

	Alternatively, you can set the guess parameter and skip the interactive
	bit. In this case it is fine, and faster to run in a standard python
	session. 

	Once a peak is selected, a stamp from the full 2D distribution, 
	with a size defined by the stamp_size parameter, is fit with a 
	2d gaussian where x and y values of the peak are the secondary and 
	primary RVs, repsectively. 
	(There is some code below to do a 2D quadratic fit and the interpolated
	 maximum below, but I'm going with the 2D gaussian in this version.)

	The alpha distributions is interpolated over with a cublic spline
	to return the flux ratio value at the peak location.

	How do I know I have the best template? 
	- I typically take a high s/n spectrum where the primary and 
	  secondary are very well separated in velocity, and run todcor
	  in the non-interactive mode over a grid of templates. The best pair 
	  should produce the highest TOODCOR peak (returned in the 'tod_vals' 
	  array). 
	  Note that the parameter space can be huge: different temperatures,
	  rotational broadening, etc. 

	Parameters
    ----------
    t_f_names: array-like
		Array of keywords for a science spectrum SAPHIRES dictionary. Output of 
		one of the saphires.io read-in functions.

	t_spec1 : python dictionary
		SAPHIRES dictionary for the science spectrum that has been prepared with 
		the utils.prepare function with a template spectrum. If using different 
		spectral templates (i.e. different temperatures/spectral types) this 
		should generally be the hotter of the two. The flux ratio (alpha) assumes
		this should be the brighter of the two, which in most cases is the hotter
		star. 

	t_spec2 : python dictionary
		SAPHIRES dictionary for the science spectrum that has been prepared with 
		the utils.prepare function with a template spectrum. If using different 
		spectral templates (i.e. different temperatures/spectral types) this 
		should generally be the cooler of the two. The flux ratio (alpha) assumes
		this should be the dimmer of the two, which in most cases is the cooler
		star. 
		If you using the same template for both stars, you do not need to 
		prepare the same science spectrum twice, t_spec1 and t_spec2 can be the
		same dictionary

	vel_width : float
		The range over which to compute the two-dimensional cross correlation.
		Larger values take longer to run. Small values could exclude a peak 
		you care about. The default value is 200 km/s. 

	stamp_size : int
		The size around the peak to fit with a gaussian. Units are in grid
		point units, i.e. velocity spacing steps. The default value is 5, but
		the ideal value depends on the velocity resolution and if you have
		done any oversampling in the utils.prepare step, best to try a few 
		values. 

	alpha_fit : bool; float
		Whether the flux ratio should be a fit parameter. If True, the flux 
		ratio is fit, and a contour plot is presented in the interactive mode
		to aid in selecting the most appropriate peak. If False, the value
		is set to 1. If a float is presented, the alpha value is set to that
		float. The default value is True.

	guess : bool; array-like
		Whether to run todcor in the interactive mode. If False, todcor is 
		run in the interactive mode where you choose the best peak. The 
		alternative options is a two element array that give a guess for the 
		primary and secondary velocities. The default value is False. 

    results : bool
    	Option to plot a zoom in of the peak stamp with the best fit peak 
    	location. A nice sanity check. The default value is True.

    text_out: bool 
    	Option to output a text file with the results of TODCOR. If True,
    	the file will have the nomenclature: 
    	[FileName]_[TempName1]_[TempName2]_todcor.dat, and will contain the 
    	RV1, RV2, flux ratio, and todcor peak values. The default value is 
    	True.

	Returns
    -------
	spectra : dictionary
		A python dictionary with the SAPHIRES architecture. The output dictionary
		will have 2 new keywords as a result of this function. And is a copy of 
		t_spec1.

		['tod_vals']  - An array with the results of the todcor fit:
					    RV1, RV2, flux ratio, TODCOR peak height.
					    Note that if you have applied a shift to your spectra 
					    with saphires.utils.apply_shift, that shift is not 
					    accounted for here -- these values are unaware of any 
					    shifts. 
		['tod_temps'] - The names of the templates used when running TODCOR
	'''

	spectra = copy.deepcopy(t_spec1)

	vel_guess=[]
	
	def press_key(event):
		if event.key == 'm':
			print('Primary Velocity:   ',event.ydata)
			print('Secondary Velocity: ',event.xdata)
			vel_guess.append(event.ydata)
			vel_guess.append(event.xdata)
	
		return vel_guess

	temp1_name = t_spec1[t_f_names[0]]['temp_name'].split('.')[0]
	temp2_name = t_spec2[t_f_names[0]]['temp_name'].split('.')[0]

	if text_out == True:
		f=open(t_f_names[0].split('.')[0]+'_'+temp1_name+'_'+temp2_name+'_todcor.dat','w')
		f.write('#TODCOR results for templates '+temp1_name+' and '+temp2_name+'\n')

	for k in range(t_f_names.size):
		
		v_spacing1 = t_spec1[t_f_names[k]]['vel_spacing']
		v_spacing2 = t_spec2[t_f_names[k]]['vel_spacing']
		
		if v_spacing1 != v_spacing2:
			print('')
			print('Different wavelength step in the two templates.')
			print('This may have adverse effects that have not been fully explored.')
			print('See documentation for possible work arounds.')
			print('')
		
		m = np.int(vel_width / v_spacing1)

		if (m/2.0 % 1) == 0:
			m=m-1

		f_s = t_spec1[t_f_names[k]]['vflux']
		f_t1 = t_spec1[t_f_names[k]]['vflux_temp']
		f_t2 = t_spec2[t_f_names[k]]['vflux_temp']

		if f_s.size < m:
			if quiet == False:
				print(t_f_names[i],t_spectra[t_f_names[i]]['w_region'])
				print("The target mask region is smaller for the m value.")
				print(w1t.size,'versus',m)
				print("You can either reduce the vel_width, or remove this order from the input,")
				print("or don't worry about it.")
				print(' ')
			spectra[t_f_names[i]]['vwave'] = 0.0	
			spectra[t_f_names[i]]['order_flag'] = 0
			continue

		c1,c1_v = utils.spec_ccf(f_s,f_t1,m,v_spacing1) # The Primary CCF
	
		c2,c2_v = utils.spec_ccf(f_s,f_t2,m,v_spacing1) # The Secondary CCF
	
		c12,c12_v = utils.spec_ccf(f_t1,f_t2,m*2+1,v_spacing1) # The Primary v Secondary CCF
	
		#This statement works when alpha_fit is given a float
		if type(alpha_fit) == float:
			alpha_in = alpha_fit
			alpha_p = (np.std(f_t2)/np.std(f_t1)) * alpha_in

		#This statement works when alpha_fit == False, in which case it is set to 1.0
		if alpha_fit == False:
			alpha_in = 1.0
			alpha_p = (np.std(f_t2)/np.std(f_t1)) * alpha_in
	
		tod = np.zeros([c1_v.size,c2_v.size])
		alpha_f = np.zeros([c1_v.size,c2_v.size])
		
		for i in range(c1_v.size):
			for j in range(c2_v.size):
				i12 = i + np.int(m/2)+1
				j12 = j + np.int(m/2)+1
	
				if alpha_fit != True:
					tod[i,j] = c1[i] + alpha_p*c2[j] / np.sqrt(1.0 + 2.0*alpha_p*c12[j12-i12] + alpha_p**2)
	
				if alpha_fit == True:
					tod[i,j] = np.sqrt((c1[i]**2 - 2.0*c1[i]*c2[j]*c12[j12-i12] + c2[j]**2) 
					                   / (1.0 - c12[j12-i12]**2))
	
					alpha_f[i,j] = ((np.std(f_t1)/np.std(f_t2)) * 
					                ((c1[i]*c12[j12-i12]-c2[j])/(c2[j]*c12[j12-i12]-c1[i])))
	
		if guess == False:
			plt.ion()

			vel_guess=[]

			fig,ax = plt.subplots(2,sharex=True,sharey=True)
		
			#A reminder that python indecies are ROW-COLUMN
			# i = primary   = 1 = row    = Y
			# j = secondary = 2 = column = X
			cs=ax[0].contour(c2_v,c1_v,tod,cmap='YlGnBu',levels=np.arange(0,np.max(tod),0.005))
			ax[0].plot(c2_v,c1_v,':',color='k')
			ax[0].set_xlabel('Secondary Velocity (km/s)')
			ax[0].set_ylabel('Primary Velocity (km/s)')
			cbar = plt.colorbar(cs,format="%3.2f",ax=ax[0])
			ax[0].set_title('Two-Dimentional CCF')

			if alpha_fit == True:
				cs=ax[1].contour(c2_v,c1_v,alpha_f,cmap='YlGnBu',levels=np.arange(0,1,0.05))
				ax[1].plot(c2_v,c1_v,':',color='k')
				ax[1].set_xlabel('Secondary Velocity (km/s)')
				ax[1].set_ylabel('Primary Velocity (km/s)')
				cbar = plt.colorbar(cs,format="%3.2f",ax=ax[1])
				ax[1].set_title(r'Flux Ratio ($\alpha$)')
		
			plt.tight_layout()
			
			print('')
			print("Press 'm' over the peak you want to fit.")
			print("Press return when done.")

			cid = fig.canvas.mpl_connect('key_press_event',press_key)
			
			wait = p_input('')
		
			fig.canvas.mpl_disconnect(cid)
		
			plt.cla()
		
			plt.close()

		else:
			vel_guess = guess
		
		#Indecies for the center of the fitting stamp location
		pg_ind=np.where(np.abs(c1_v - vel_guess[0]) == np.min(np.abs(c1_v - vel_guess[0])))[0][0]
		sg_ind=np.where(np.abs(c2_v - vel_guess[1]) == np.min(np.abs(c2_v - vel_guess[1])))[0][0]
		
		#The row-column nature of python arrays when ploting has things looking weird here but the 
		#current set us is returning the correct answers.
		stamp = tod[pg_ind-stamp_size:pg_ind+stamp_size+1,sg_ind-stamp_size:sg_ind+stamp_size+1]

		c1_v_cut = c1_v[pg_ind-stamp_size:pg_ind+stamp_size+1]
		c2_v_cut = c2_v[sg_ind-stamp_size:sg_ind+stamp_size+1]
		c2_v_cutm,c1_v_cutm = np.meshgrid(c2_v_cut,c1_v_cut)

		#---------------------- Raw Maximum ---------------------------------------
		#stamp_max_ind = np.where(stamp == np.max(stamp))
		#c1_max = c1_v_cut[stamp_max_ind[0][0]]
		#c2_max = c2_v_cut[stamp_max_ind[1][0]]
		#
		#print c2_max,c1_max

		#--------------- 2D Gaussian Fit ------------------------------------------
		fit_guess = [np.max(stamp), vel_guess[1], vel_guess[0], 7, 12, 90, 0]
		td_fit, td_cov = curve_fit(utils.td_gaussian, (c2_v_cutm,c1_v_cutm), stamp.ravel(), 
		                           p0=fit_guess, maxfev=20000,
		                           bounds = ((0,np.min(c2_v_cut),np.min(c1_v_cut),0,0,0,0),
		                           			(10,np.max(c2_v_cut),np.max(c1_v_cut),100,100,360,10)))
		fit_gauss1d = utils.td_gaussian((c2_v_cutm,c1_v_cutm),*td_fit)
		fit_gauss = fit_gauss1d.reshape(c2_v_cut.size,c1_v_cut.size)

		#-------------- 2D Quadratic Fit ------------------------------------------
		#qd_fit, qd_cov = curve_fit(td_quad, (c2_v_cutm,c1_v_cutm), stamp.ravel(), maxfev=20000)
		#a,b,c,d,e,g = qd_fit
		#rv2_q = -(2*b*d - e*c)/(4*a*b - c**2)
		#rv1_q = -(2*a*e - d*c)/(4*a*b - c**2)
		#fit_quad1d = td_quad((c2_v_cutm,c1_v_cutm),*qd_fit)
		#fit_quad = fit_quad1d.reshape(c2_v_cut.size,c1_v_cut.size)


		#------------ 2D Interpolation Max -------------------------------------
		#f_stamp = interpolate.interp2d(c1_v_cut,c2_v_cut,stamp,kind='cubic')

		#c1_v_oversample = np.linspace(np.min(c1_v_cut),np.max(c1_v_cut),c1_v_cut.size*100)
		#c2_v_oversample = np.linspace(np.min(c2_v_cut),np.max(c2_v_cut),c2_v_cut.size*100)

		#stamp_over = f_stamp(c1_v_oversample,c2_v_oversample)
		#stamp_over_max_ind = np.where(stamp_over == np.max(stamp_over))
		#rv1_imax = c1_v_oversample[stamp_over_max_ind[0][0]]
		#rv2_imax = c2_v_oversample[stamp_over_max_ind[1][0]]


		#Peak Height and Alpha value determinations
		if alpha_fit == True:
			alph_stamp = alpha_f[pg_ind-stamp_size:pg_ind+stamp_size+1,sg_ind-stamp_size:sg_ind+stamp_size+1]
			f_alph_stamp = interpolate.interp2d(c1_v_cut,c2_v_cut,alph_stamp,kind='cubic')
			
			f_ratio = f_alph_stamp(td_fit[2],td_fit[1])[0]

		else:
			f_ratio = alpha_in

		f_stamp = interpolate.interp2d(c1_v_cut,c2_v_cut,stamp,kind='cubic')
		peak = f_stamp(td_fit[2],td_fit[1])[0]


		spectra[t_f_names[k]]['tod_vals'] = np.array([td_fit[2],td_fit[1],f_ratio,peak])
		spectra[t_f_names[k]]['tod_temps'] = np.array([temp1_name,temp2_name])

		if results == True:
			fig,ax = plt.subplots(1)
			
			ax.contour(c2_v_cut,c1_v_cut,stamp,100,colors='white')
			ax.imshow(stamp,cmap='binary',extent=[np.min(c2_v_cut),np.max(c2_v_cut),
			          							  np.min(c1_v_cut),np.max(c1_v_cut)])

			ax.plot(vel_guess[1],vel_guess[0],'ro',label='guess')
			ax.plot(td_fit[1],td_fit[2],'o',color='orange',label='2D Gaussian Fit')
			#ax.plot(rv2_q,rv1_q,'go',label='Quadratic Fit')
			#ax.plot(c2_max,c1_max,'o',color='grey',label='Raw Max')
			#ax.plot(rv2_imax,rv1_imax,'o',color='pink',label='Interp Max')

			print('Guess:   ',vel_guess[0],vel_guess[1])
			#print 'Raw Max: ',c2_max,c1_max
			print('2D Gauss:',td_fit[2],td_fit[1])
			#print 'Quad:    ',rv2_q,rv1_q
			#print 'Interp:  ',rv2_imax,rv1_imax

			ax.legend()
			ax.set_xlabel('Secondary Velocity (km/s)')
			ax.set_ylabel('Primary Velocity (km/s)')
		
			print('')
			print('Press return when done.')

			wait = p_input('')
			
			plt.cla()
			
			plt.close()

		if text_out == True:
			if k == 0:
				f.write('#Spectrum\tRV1\tRV2\tFlux Ratio\tPeak Height\n')
			f.write(np.str(t_f_names[k])+'\t'+
					np.str(np.round(td_fit[2],5))+'\t'+
					np.str(np.round(td_fit[1],5))+'\t'+
					#np.str(rv1_q)+'\t'+
					#np.str(rv2_q)+'\t'+
					#np.str(rv1_imax)+'\t'+
					#np.str(rv2_imax)+'\t'+
					np.str(np.round(f_ratio,3))+'\t'+
					np.str(np.round(peak,3))+'\n')

		vel_guess = []

	if text_out == True:
		f.close()

	plt.ioff()
	return spectra


