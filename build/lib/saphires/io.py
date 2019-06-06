'''
############################ SAPHIRES io ##############################
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
A collection of functions that read in and out put spectra in the 
SAPHIRES data format.
'''
# ---- Standard Library
import sys
# ----

# ---- Third Party
import pickle as pkl
import numpy as np
import astropy.io.fits as pyfits
# ---- 

# ---- Project
from saphires import utils
from saphires.extras import bspline_acr as bspl
# ----

py_version = sys.version_info.major
if py_version == 3:
	nplts = 'U'	#the numph letter for a string
if py_version == 2:
	nplts = 'S' #the numph letter for a string

#For python 3
#spec = pkl.load(open('spec.p','rb'),encoding='bytes')

#For python 2
#spec = pkl.load(open('spec.p','rb'))

#Science spectra
#0  - ['nflux'] 		- native flux array (inverted)						
#1  - ['nwave'] 		- native wavelength array							
#2  - ['ndw'] 			- native wavelength spacing value					
#3  - ['wav_cent'] 		- Order Central Wavelength							
#4  - ['vflux'] 		- resampled flux array (inverted)					
#5  - ['vwave'] 		- resampled wavelength array							
#6  - ['vflux_temp'] 	- resampled template flux array (inverted)			
#7  - ['vel'] 			- velocity array to be used with the BF				
#8  - ['w_region'] 		- wavelength region									
#9  - ['temp_name'] 	- template name										
#10 - ['w_region_temp'] - template wavelength region							
#11 - ['bf'] 			- broadening funciton								
#12 - ['sbf'] 			- smoothed broadening function 						
#13 - ['bf_fit_params'] - gaussian fit parameters							
#14 - ['rv_shift'] 		- initial rv shift (used if you bf_compute_iter)		
#15 - ['order_flag'] 	- order flag											
#16 - ['w_region_iter'] - iter wavelength region								
#17 - ['ccf_fit_params']- fit paramters of CCF								

#Template spectra
#[0] - ['nflux'] native flux array (inverted)
#[1] - ['nwave'] native wavelength array
#[2] - ['ndw'] native wavelength spacing value
#[3] - ['wav_cent'] Order Central Wavelength
#[4] - ['w_region'] wavelength region

def read_pkl(spectra_list,temp=False,combine_all=True,norm=True,w_mult=1.0,
             trim_style='clip',norm_w_width=200.0,dk_wav='wav',dk_flux='flux'):
	'''
	A function to read in a target spectrum, or list of target spectra, from 
	a pickle dictionary with specific keywords, and put them into the 
	SAPHIRES data structure.
	
	Required Keywords:
	- Wavelength array keyword is specified by the dk_wav paramter. It must
	  contain a single or multidimensional (i.e. multiple orders) array. 
	  The wavelength array is assumed to be in angstroms.
	- Flux array keyword is specified by the dk_flux parameter. It must 
	  contain flux values corresponding to the wavelength array(s).
	Both must have the same dimenisionality and length
	
	The "data structure" is nothing special, just a set of nested dictionaries,
	but it is a simple way to keep all of the relevant information in one place
	that is, hopefully, straight forward. 

	The returned data structure has two parts:
	1) A list of dictionary keywords. They have the following form:
	   "file_name[order][wavelength range]"
	2) A dictionary. For each of the keywords above there is a nested
	   dictionary with the following form:
	   ['nflux'] 		- native flux array (inverted)
	   ['nwave'] 		- native wavelength array
	   ['ndw'] 			- native wavelength spacing
	   ['wav_cent'] 	- order central wavelength
	   ['w_region'] 	- wavelength region
	   ['rv_shift'] 	- initial rv shift
	   ['order_flag'] 	- order flag	

	If you specify that you are reading in a template, there will only be
	a dictionary with the following keywords (it is not nested with spectral
	order descriptions):
	   ['nflux'] 		- native flux array (inverted)
	   ['nwave'] 		- native wavelength array
	   ['ndw'] 			- native wavelength spacing
	   ['wav_cent'] 	- order central wavelength
	   ['w_region'] 	- wavelength region
	   ['temp_name'] 	- template file name
	
	Additional keywords and data are added to the nested dictionaries by 
	other SAPHIRES functions, e.g.:
	['vflux'] 			- resampled flux array (inverted)					
	['vwave'] 			- resampled wavelength array							
	['vflux_temp'] 		- resampled template flux array (inverted)			
	['vel'] 			- velocity array to be used with the BF or CCF		
	['temp_name'] 		- template name										
	['bf'] 				- broadening funciton								
	['sbf'] 			- smoothed broadening function 						
	['bf_fit_params'] 	- gaussian fit parameters							
	More details on this is various functions. 

	Many of the spectral analysis techniques (cross correlation, broadening 
	function, etc) work with a inverted spectrum, i.e. the continuum is at
	zero and absorption lines extend upward - so all of the flux arrays are 
	inverted in this way.

	Parameters
    ----------
	spectra_list : str
		The file to be read in. Can be the name of a single pickle file, or
		a list of spectra/spectral regions.
		- Single file -- must end in a '.p' or '.pkl' and have the format
		  described above.
		- List of spectra -- text file that must end in '.ls', '.dat', '.txt'.
		  The input file must have the following format:
		  filename order wavelength_range
		  Some examples:
		  spec.p 0 * 
		  - (reads in the entire wavelenth range as one order)
		  spec.p 0 5200-5300,5350-5400
		  spec.p 0 5400-5600
		  - (splits the single wavelength array in to two "orders" that are 
		  	 each 200 A wide. The first has a 50 A gap.)
		  Notes on wavelength_range: 
		  	Must have the general form "w1-w2,w3-w4" where '-' symbols 
		  	includes the wavelength region, ',' symbols excludes them. There 
		  	can be as many regions as you want, as long as it ends with an
		  	inclusive region (i.e. cannot end with a comma or dash). 
		  	Wavelength values must ascend left to right. 
		I would highly recommend using the specified wavelength regions. 
		There will inevitably be a part of the spectrum (edge effect, CR,
		intrinsic emission line) that you don't want to include in your
		calculation. And, if you remove overlapping regions, you can set the
		'combine_all' parameter to True to create a stiched order, which can 
		be useful.

	temp : bool
		Tell the function whether this is a template spectrum. Science and 
		template spectra have slightly different formats, namely, that 
		template spectra are not nested dictionaries with keywords that 
		describe spectral orders -- they are just a standard dictionary.
		When this parameter is True, only a dictionary is output, without 
		an array of header keywords. 
		Both cases read in data in the same way.
		The default value is False.

	combine_all : bool
		Option to stitch together all spectral orders. This is useful generally,
		but especially for low-S/N spectra where any given order could give you 
		BF or CCF results that are trash. The default value is 'True'.
		IMPORTANT NOTE: Spectra are currently stitched in the simplist way 
		possible. Use the defined spectral windows capabilties of the 
		'spectra_list' parameter to avoid overlapping regions from order to 
		order. 

	norm : bool
		Option to continuum normalize the input spectrum. Default is True.

	norm_w_width : float
		If the 'norm' paramter is 'True', the parameter set the width of the
		normalization window. The default is 200 Angstroms, which does a 
		decent job in a variety of situtaions. 

	w_mult : float
		Value to multiply the wavelength array. This is used to convert the 
		input wavelength array to Angstroms if it is not already. The default 
		is 1, assuming the wavelength array is alreay in Angstroms. 

	trim_style : str, options: 'clip', 'lin', 'spl'
		If a wavelength region file is input in the 'spectra_list' parameter, 
		this parameter describes how gaps are dealt with. 
		- If 'clip', unused regions will be left as gaps.
		- If 'lin', unused regions will be linearly interpolated over.
		- If 'spl', unused regions will be interpolated over with a cubic 
		  spline. You probably don't want to use this one.

	dk_wav : str
		Dictionary keyword for the wavelength array. Default is 'wav'

	dk_flux : str
		Dictionary keyword for the flux array. Default is 'flux'
		
    Returns
    -------
	tar : array-like
		List of dictionary keywords, described above.

	tar_spec : dictionary
		SAPHIRES dictionary, described above. 
	'''

	in_type = spectra_list.split('.')[-1]

	direct = ['p','pkl']
	ord_list = ['ls','txt','dat']

	if in_type in direct:
		t_f_names = np.array([spectra_list])
		order = np.array([0])
		w_range = np.array(['*'])

	if in_type in ord_list:
		t_f_names,order,w_range = np.loadtxt(spectra_list,unpack=True,
		                                     dtype=nplts+'10000,i,'+nplts+'10000')
		if (t_f_names.size == 1): 
			t_f_names=np.array([t_f_names])
			order=np.array([order])
			w_range=np.array([w_range])

	if ((in_type not in direct) & (in_type not in ord_list)):
		print('Input file in wrong format.')
		print("If a single pickle dictionary file, must end in '.p' or '.pkl'.")
		print("If a input text file, must end in '.ls', '.dat', or '.txt'.")
		return 0,0

	#Dictionary for output spectra
	t_spectra={}

	t_f_names_out=np.empty(0,dtype=nplts+'100')

	for i in range(np.unique(t_f_names).size):
		#-------------- READ IN -------------------------
		if py_version == 2:
			pic_dic = pkl.load(open(t_f_names[i],'rb'))
		if py_version == 3:
			pic_dic = pkl.load(open(t_f_names[i],'rb'),encoding='latin')

		keys = list(pic_dic.keys())

		if dk_wav not in keys:
			print("The wavelength array dictionary keyword specified, '"+dk_wav+"'")
			print("was not found.")
			return 0,0
		if dk_flux not in keys:
			print("The flux array dictionary keyword specified, '"+dk_flux+"'")
			print("was not found.")
			return 0,0

		if (pic_dic[dk_wav].ndim == 1) & (in_type in direct):
			n_orders = 1

		if (pic_dic[dk_wav].ndim == 1) & (in_type in ord_list):
			n_orders = order.size

		if (pic_dic[dk_wav].ndim > 1) & (in_type in direct):
			n_orders=pic_dic[dk_wav].shape[0]

		if (pic_dic[dk_wav].ndim > 1) & (in_type in ord_list):
			n_orders = order.size

		for j in range(n_orders):
			if in_type in ord_list:
				j_ind=order[j]
			if in_type in direct:
				j_ind = j

			if pic_dic[dk_wav].ndim == 1:
				t_flux = pic_dic[dk_flux]
				t_w = pic_dic[dk_wav]
			else:
				t_flux = pic_dic[dk_flux][j_ind]
				t_w = pic_dic[dk_wav][j_ind]

			if in_type in direct:
				w_range_out = np.str(np.int(np.min(t_w)))+'-'+np.str(np.int(np.max(t_w)))
			if in_type in ord_list:
				if w_range[j] == '*':
					w_range_out = np.str(np.int(np.min(t_w)))+'-'+np.str(np.int(np.max(t_w)))
				else:
					w_range_out = w_range[j]

		#------------------------------------------------------

			t_w = t_w*w_mult

			t_w, t_flux = utils.spec_trim(t_w,t_flux,w_range_out,'*',trim_style=trim_style)
		
			if norm == True:
				t_flux = t_flux / np.median(t_flux)
				t_flux = utils.cont_norm(t_w,t_flux,w_width=norm_w_width)

			t_flux = 1.0 - t_flux

			w_min=np.int(np.min(t_w))
			w_max=np.int(np.max(t_w))

			t_dw = np.median(t_w - np.roll(t_w,1))

			t_f_names_out=np.append(t_f_names_out,
			                        t_f_names[i]+'['+np.str(j_ind)+']['+np.str(w_min)+'-'+
			                        np.str(w_max)+']')

			t_spectra[t_f_names_out[-1]]={'nflux': t_flux,
										  'nwave': t_w,
										  'ndw': t_dw,
										  'wav_cent': np.mean(t_w),
										  'w_region': w_range_out,
										  'rv_shift': 0.0,
										  'order_flag': 1}

	if ((combine_all == True) | (temp == True)):
		w_all = np.empty(0)
		flux_all = np.empty(0)

		for i in range(t_f_names_out.size):
			w_all = np.append(w_all,t_spectra[t_f_names_out[i]]['nwave'])
			flux_all = np.append(flux_all,t_spectra[t_f_names_out[i]]['nflux'])
			
			if t_f_names_out.size > 1:
				if i == 0:
					w_range_all = t_spectra[t_f_names_out[i]]['w_region']+','
				if ((i > 0) & (i<t_f_names_out.size-1)):
					w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']+','
				if i == t_f_names_out.size-1:
					w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']
			if t_f_names_out.size == 1:
					w_range_all = t_spectra[t_f_names_out[i]]['w_region']

		w_min=np.int(np.min(w_all))
		w_max=np.int(np.max(w_all))

		t_dw = np.median(w_all - np.roll(w_all,1))

		t_spectra['Combined']={'nflux': flux_all,
							   'nwave': w_all,
							   'ndw': t_dw,
							   'wav_cent': np.mean(w_all),
							   'w_region': w_range_all,
							   'rv_shift': 0.0,
							   'order_flag': 1}

		t_f_names_out=np.append(t_f_names_out,'Combined')

	if temp == True:
		temp_spectra={'nflux': flux_all,
					  'nwave': w_all,
					  'ndw': t_dw,
					  'wav_cent': np.mean(w_all),
					  'w_region': w_range_all,
					  'temp_name': t_f_names[0]}
		return temp_spectra

	return t_f_names_out,t_spectra


def read_fits(spectra_list,temp=False,w_mult=1.0,combine_all=True,norm=True,
                norm_w_width=200.0,trim_style='clip'):
	'''
	A function to read in a target spectrum, or list of target spectra, from 
	an IRAF friendly fits file with a single order, and put them into the 
	SAPHIRES data structure.

	IRAF friendly in this context means that the fits file contrains a flux 
	array and the headers have keywords CRVAL1, CDELT1, and potentially, LTV1,
	that define the wavelength array. If you don't have these keywords, it will
	not work. 
	
	The "data structure" is nothing special, just a set of nested dictionaries,
	but it is a simple way to keep all of the relevant information in one place
	that is, hopefully, straight forward. 

	The returned data structure has two parts:
	1) A list of dictionary keywords. They have the following form:
	   "file_name[order][wavelength range]"
	2) A dictionary. For each of the keywords above there is a nested
	   dictionary with the following form:
	   ['nflux'] 		- native flux array (inverted)
	   ['nwave'] 		- native wavelength array
	   ['ndw'] 			- native wavelength spacing
	   ['wav_cent'] 	- order central wavelength
	   ['w_region'] 	- wavelength region
	   ['rv_shift'] 	- initial rv shift
	   ['order_flag'] 	- order flag				
	
	If you specify that you are reading in a template, there will only be
	a dictionary with the following keywords (it is not nested with spectral
	order descriptions):
	   ['nflux'] 		- native flux array (inverted)
	   ['nwave'] 		- native wavelength array
	   ['ndw'] 			- native wavelength spacing
	   ['wav_cent'] 	- order central wavelength
	   ['w_region'] 	- wavelength region
	   ['temp_name'] 	- template file name

	Additional keywords and data are added to the nested dictionaries by 
	other SAPHIRES functions, e.g.:
	['vflux'] 			- resampled flux array (inverted)					
	['vwave'] 			- resampled wavelength array							
	['vflux_temp'] 		- resampled template flux array (inverted)			
	['vel'] 			- velocity array to be used with the BF or CCF		
	['temp_name'] 		- template name										
	['bf'] 				- broadening funciton								
	['sbf'] 			- smoothed broadening function 						
	['bf_fit_params'] 	- gaussian fit parameters							
	More details on this is various functions. 

	Many of the spectral analysis techniques (cross correlation, broadening 
	function, etc) work with a inverted spectrum, i.e. the continuum is at
	zero and absorption lines extend upward - so all of the flux arrays are 
	inverted in this way.

	Parameters
    ----------
	spectra_list : str
		The file to be read in. Can be the name of a single pickle file, or
		a list of spectra/spectral regions.
		- Single file -- must end in '.fits' or '.fit'.
		- List of spectra -- text file that must end in '.ls', '.dat', '.txt'.
		  The input file must have the following format:
		  filename order wavelength_range
		  Some examples:
		  spec.fits 0 * 
		  - (reads in the entire wavelenth range as one order)
		  spec.p 0 5200-5300,5350-5400
		  spec.p 0 5400-5600
		  - (splits the single wavelength array in to two "orders" that are 
		  	 each 200 A wide. The first has a 50 A gap.)
		  Notes on wavelength_range: 
		  	Must have the general form "w1-w2,w3-w4" where '-' symbols 
		  	includes the wavelength region, ',' symbols excludes them. There 
		  	can be as many regions as you want, as long as it ends with an
		  	inclusive region (i.e. cannot end with a comma or dash). 
		  	Wavelength values must ascend left to right. 
		I would highly recommend using the specified wavelength regions. 
		There will inevitably be a part of the spectrum (edge effect, CR,
		intrinsic emission line) that you don't want to include in your
		calculation. And, if you remove overlapping regions, you can set the
		'combine_all' parameter to True to create a stiched order, which can 
		be useful.

	temp : bool
		Tell the function whether this is a template spectrum. Science and 
		template spectra have slightly different formats, namely, that 
		template spectra are not nested dictionaries with keywords that 
		describe spectral orders -- they are just a standard dictionary.
		When this parameter is True, only a dictionary is output, without 
		an array of header keywords. 
		Both cases read in data in the same way.
		The default value is False.

	combine_all : bool
		Option to stitch together all spectral orders. This is useful generally,
		but especially for low-S/N spectra where any given order could give you 
		BF or CCF results that are trash. The default value is 'True'.
		IMPORTANT NOTE: Spectra are currently stitched in the simplist way 
		possible. Use the defined spectral windows capabilties of the 
		'spectra_list' parameter to avoid overlapping regions from order to 
		order. 

	norm : bool
		Option to continuum normalize the input spectrum. Default is True.

	norm_w_width : float
		If the 'norm' paramter is 'True', the parameter set the width of the
		normalization window. The default is 200 Angstroms, which does a 
		decent job in a variety of situtaions. 

	w_mult : float
		Value to multiply the wavelength array. This is used to convert the 
		input wavelength array to Angstroms if it is not already. The default 
		is 1, assuming the wavelength array is alreay in Angstroms. 

	trim_style : str, options: 'clip', 'lin', 'spl'
		If a wavelength region file is input in the 'spectra_list' parameter, 
		this parameter describes how gaps are dealt with. 
		- If 'clip', unused regions will be left as gaps.
		- If 'lin', unused regions will be linearly interpolated over.
		- If 'spl', unused regions will be interpolated over with a cubic 
		  spline. You probably don't want to use this one.
		
    Returns
    -------
	tar : array-like
		List of dictionary keywords, described above.

	tar_spec : dictionary
		SAPHIRES dictionary, described above. 
	'''
	#Target/Science Spectra Read in the name of files. 

	in_type = spectra_list.split('.')[-1]

	direct = ['fit','fits']
	ord_list = ['ls','txt','dat']

	if in_type in direct:
		t_f_names = np.array([spectra_list])
		order = np.array([0])
		w_range = np.array(['*'])

	if in_type in ord_list:
		t_f_names,order,w_range = np.loadtxt(spectra_list,unpack=True,
		                                     dtype=nplts+'10000,i,'+nplts+'10000')
		if (t_f_names.size == 1): 
			t_f_names=np.array([t_f_names])
			order=np.array([order])
			w_range=np.array([w_range])

	if ((in_type not in direct) & (in_type not in ord_list)):
		print('Input file in wrong format.')
		print("If a single fits file, must end in '.fits' or '.fit'.")
		print("If a input text file, must end in '.ls', '.dat', or '.txt'.")
		return 0,0

    #Dictionary for output spectra
	t_spectra={}
	t_f_names_out=np.empty(0,dtype=nplts+'100')

	#Read in Target spectra
	for i in range(t_f_names.size):
		t_hdulist=pyfits.open(t_f_names[i])
		t_flux=t_hdulist[0].data
		t_w0=np.float(t_hdulist[0].header['CRVAL1'])
		t_dw=np.float(t_hdulist[0].header['CDELT1'])

		if 'LTV1' in t_hdulist[0].header:
			t_shift=np.float(t_hdulist[0].header['LTV1'])
			t_w0=t_w0-t_shift*t_dw

		t_w0=t_w0 * w_mult
		t_dw=t_dw * w_mult

		#linear wavelength array
		t_w=np.arange(t_flux.size)*t_dw+t_w0

		t_w,t_flux = utils.spec_trim(t_w,t_flux,w_range[i],'*',trim_style=trim_style)
		#t_flux = t_flux[trim_mask]
		
		if norm == True:
			t_flux = t_flux / np.median(t_flux)
			t_flux = utils.cont_norm(t_w,t_flux,w_width=norm_w_width)

		#inverted spectrum
		t_flux=1.0-t_flux

		if in_type in direct:
				w_range_out = np.str(np.int(np.min(t_w)))+'-'+np.str(np.int(np.max(t_w)))
		if in_type in ord_list:
			if w_range[i] == '*':
				w_range_out = np.str(np.int(np.min(t_w)))+'-'+np.str(np.int(np.max(t_w)))
			else:
				w_range_out = w_range[i]


		t_f_names_out=np.append(t_f_names_out,
			                        t_f_names[i]+'['+np.str(order[i])+']['+np.str(np.int(np.min(t_w)))+'-'+
			                        np.str(np.int(np.max(t_w)))+']')

		t_spectra[t_f_names_out[-1]]={'nflux': t_flux,
									  'nwave': t_w,
									  'ndw': t_dw,
									  'wav_cent': np.mean(t_w),
									  'w_region': w_range_out,
									  'rv_shift': 0.0,
									  'order_flag': 1}

	if ((combine_all == True) | (temp == True)):
		w_all = np.empty(0)
		flux_all = np.empty(0)

		for i in range(t_f_names.size):
			w_all = np.append(w_all,t_spectra[t_f_names_out[i]]['nwave'])
			flux_all = np.append(flux_all,t_spectra[t_f_names_out[i]]['nflux'])
			
			if t_f_names_out.size > 1:
				if i == 0:
					w_range_all = t_spectra[t_f_names_out[i]]['w_region']+','
				if ((i > 0) & (i<t_f_names_out.size-1)):
					w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']+','
				if i == t_f_names_out.size-1:
					w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']
			if t_f_names_out.size == 1:
					w_range_all = t_spectra[t_f_names_out[i]]['w_region']

		w_min=np.int(np.min(w_all))
		w_max=np.int(np.max(w_all))

		t_dw = np.median(w_all - np.roll(w_all,1))

		t_spectra['Combined']={'nflux': flux_all,
							   'nwave': w_all,
							   'ndw': t_dw,
							   'wav_cent': np.mean(w_all),
							   'w_region': w_range_all,
							   'rv_shift': 0.0,
							   'order_flag': 1}

		t_f_names_out=np.append(t_f_names_out,'Combined')

	if temp == True:
		temp_spectra={'nflux': flux_all,
					  'nwave': w_all,
					  'ndw': t_dw,
					  'wav_cent': np.mean(w_all),
					  'w_region': w_range_all,
					  'temp_name': t_f_names[0]}
		return temp_spectra

	return t_f_names_out,t_spectra








