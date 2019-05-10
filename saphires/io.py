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

Description:
A collection of functions that read in and out put spectra in the 
SAPHIRES data format.
'''

# ---- Standard Library
import sys
# ----

# ---- Third Party
import pickle as pkl
import numpy as np
# ---- 

# ---- Project
from saphires import utils
from saphires.extras import bspline_acr as bspl
# ----

py_version = sys.version_info.major
if py_version == 3:
	nplts = 'U'	
if py_version == 2:
	nplts = 'S'

#For python 3
#spec = pkl.load(open('lte05900-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_2800-11000.p','rb'),encoding='bytes')

#For python 2
#spec = pkl.load(open('lte05900-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_2800-11000.p','rb'))


def target_pkl(spectra_list,w_file=None,combine_all=True,norm=True,w_mult=1.0,trim_style='clip',norm_w_width=200.0):
	'''
	A function to read in a target spectrum, or list of target spectra
	with which to compute a broadening function. The template should be
	continuum nornalized.

	This function is intended for use with pickled files that are a single
	order. The opened pickle should be a dictionary for a single spectrum
	with the a wavelength array in angstroms under the keyword "wav" and a 
	flux array under the keyword "flux".

	"spectra_list" is a test file with one name of a pickle file per line
	in the first column. In the second column you can define a range over
	which to compute the broadening function. The convention is, 
	"w1-w2,w3-w4", and you can have as many as you like.
	If you want to use the entire wavelenth range, put a "*" in the second 
	column.

	Set "tar_wnm" if the fits spacing are in nanometers instead of angstroms.
	The program assume the native wavelength spacing is in angstroms. 

	This function fill return a list of the input file names as an array and
	a dictionary where the name of each of the input files is a dictionary 
	keyword. Each keyword will have following arrays assocated with it:

	[0] - native flux array (inverted)						'nflux': t_flux,
	[1] - native wavelength array							'nwave': t_w,
	[2] - native wavelength spacing value					'ndw': t_dw,
	[3] - Order Central Wavelength							'wav_cent': np.mean(t_w),
	[4] - resampled flux array (inverted)					'vflux': 0,
	[5] - resampled wavelength array						'vwave': 0,
	[6] - resampled template flux array (inverted)			'vflux_temp': 0,
	[7] - velocity array to be used with the BF				'vel': 0,
	[8] - wavelength region									'w_region': w_range_out,
	[9] - template name										'temp_name': 0,
	[10]- template wavelength region						'w_region_temp': 0,
	[11]- broadening funciton								'bf': 0,
    [12]- smoothed broadening function 						'sbf': 0,
    [13]- gaussian fit parameters							'bf_fit_params':0,
	[14]- initial rv shift (used if you bf_compute_iter)	'rv_shift': 0,
	[15]- order flag										'order_flag': 0,
	[16]- iter wavelength region							'w_region_iter': '*',
	[17]- fit paramters of CCF								'ccf_fit_params': 0}

	Most of these will be empty at the return of the this function but will 
	be filled by the bf_compute and bf_analysis functions.
	Idecies 4, 5, 6, 7, 9, 10 and 11 will be zeros.
	'''

	if w_file == None:
		if (spectra_list[-2:] == 'ls') | (spectra_list[-2:] == 'xt'):
			t_f_names=np.loadtxt(spectra_list,unpack=True,dtype='U10000')
		if (spectra_list[-2:] == '.p'):
			t_f_names = np.array(spectra_list,dtype=str)

		if (t_f_names.size == 1): 
			t_f_names=np.array([t_f_names])

	if w_file != None:
		t_f_names,order,w_range = np.loadtxt(w_file,unpack=True,dtype=nplts+'10000,i,'+nplts+'10000')

		if (t_f_names.size == 1): 
			t_f_names=np.array([t_f_names])
			order=np.array([order])
			w_range=np.array([w_range])

	#Dictionary for output spectra
	t_spectra={}

	t_f_names_out=np.empty(0,dtype='U100')

	for i in range(np.unique(t_f_names).size):
		if py_version == 2:
			pic_dic = pkl.load(open(t_f_names[i],'rb'))
		if py_version == 3:
			pic_dic = pkl.load(open(t_f_names[i],'rb'),encoding='latin')

		if (pic_dic['wav'].ndim == 1) & (w_file == None):
			n_orders = 1

		if (pic_dic['wav'].ndim == 1) & (w_file != None):
			n_orders = order.size

		if (pic_dic['wav'].ndim > 1) & (w_file == None):
			n_orders=pic_dic['wav'].shape[0]

		if (pic_dic['wav'].ndim > 1) & (w_file != None):
			n_orders = order.size

		for j in range(n_orders):
			if w_file != None:
				j_ind=order[j]
			else:
				j_ind = j
			if pic_dic['wav'].ndim == 1:
				t_flux = pic_dic['flux']
				t_w = pic_dic['wav']
			else:
				t_flux = pic_dic['flux'][j_ind]
				t_w = pic_dic['wav'][j_ind]

			t_w = t_w*w_mult

			if w_file == None:
				w_range_out = np.str(np.int(np.min(t_w)))+'-'+np.str(np.int(np.max(t_w)))
			if w_file != None:
				w_range_out = w_range[j]

			t_w, t_flux = utils.spec_trim(t_w,t_flux,w_range_out,'*',trim_style=trim_style)
			#t_flux = t_flux[trim_mask]
		
			t_flux = t_flux / np.median(t_flux)

			if norm == True:
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
										  'vflux': 0,
										  'vwave': 0,
										  'vflux_temp': 0,
										  'vel': 0,
										  'w_region': w_range_out,
										  'temp_name': 0,
										  'w_region_temp': 0,
										  'bf': 0,
										  'sbf': 0,
										  'bf_fit_params':0,
										  'rv_shift': 0,
										  'order_flag': 0,
										  'w_region_iter': '*',
										  'ccf_fit_params': 0}

	if combine_all == True:
		w_all = np.empty(0)
		flux_all = np.empty(0)

		for i in range(t_f_names_out.size):
			w_all = np.append(w_all,t_spectra[t_f_names_out[i]]['nwave'])
			flux_all = np.append(flux_all,t_spectra[t_f_names_out[i]]['nflux'])
			
			if i == 0:
				w_range_all = t_spectra[t_f_names_out[i]]['w_region']+','
			if ((i > 0) & (i<t_f_names_out.size-1)):
				w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']+','
			if i == t_f_names_out.size-1:
				w_range_all = w_range_all+t_spectra[t_f_names_out[i]]['w_region']

		w_min=np.int(np.min(w_all))
		w_max=np.int(np.max(w_all))

		t_dw = np.median(w_all - np.roll(w_all,1))

		t_spectra['Combined']={'nflux': flux_all,
							   'nwave': w_all,
							   'ndw': t_dw,
							   'wav_cent': np.mean(w_all),
							   'vflux': 0,
							   'vwave': 0,
							   'vflux_temp': 0,
							   'vel': 0,
							   'w_region': w_range_all,
							   'temp_name': 0,
							   'w_region_temp': 0,
							   'bf': 0,
							   'sbf': 0,
							   'bf_fit_params':0,
							   'rv_shift': 0,
							   'order_flag': 0,
							   'w_region_iter': '*',
							   'ccf_fit_params': 0}

		t_f_names_out=np.append(t_f_names_out,'Combined')

	return t_f_names_out,t_spectra