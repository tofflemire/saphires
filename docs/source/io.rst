=========================================================================================================================================
read_pkl(spectra_list,temp=False,combine_all=True,norm=True,w_mult=1.0,trim_style='clip',norm_w_width=200.0,dk_wav='wav',dk_flux='flux'):
=========================================================================================================================================
A function to read in a target spectrum, list of target spectra, or
template spectrum from a pickle dictionary with specific keywords, and
put them into the SAPHIRES data structure.

Required Keywords:
- Wavelength array keyword is specified by the dk_wav paramter. It must contain a single or multidimensional (i.e. multiple orders) array. The wavelength array is assumed to be in angstroms.

- Flux array keyword is specified by the dk_flux parameter. It must contain flux values corresponding to the wavelength array(s). Both must have the same dimenisionality and length

The SAPHIRES "data structure" is nothing special, just a set of nested
dictionaries, but it is a simple way to keep all of the relevant
nformation in one place that is, hopefully, straight forward.

The returned data structure has two parts:

1) A list of dictionary keywords. They have the following form: "file_name[order][wavelength range]"
2) A dictionary. For each of the keywords above there is a nested dictionary with the following form:
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
  ['vflux'] 		- resampled flux array (inverted)

  ['vwave'] 		- resampled wavelength array

  ['vflux_temp'] 	- resampled template flux array (inverted)

  ['vel'] 		- velocity array to be used with the BF or CCF

  ['temp_name'] 	- template name

  ['bf'] 			- broadening funciton

  ['bf_smooth']	- smoothed broadening function

  ['bf_fits'] 	- gaussian fit parameters

More details on these are provided in other various functions.

Many of the spectral analysis techniques (cross correlation, broadening
function, etc) work with a inverted spectrum, i.e. the continuum is at
zero and absorption lines extend upward - so all of the flux arrays are
inverted in this way.

**********
Parameters
**********
spectra_list : str

The file to be read in. Can be the name of a single pickle file, or
a list of spectra/spectral regions.

- Single file -- must end in a '.p' or '.pkl' and have the format described above.

If you are using python 3, you can read in a pickle that was made
in python 3 or 2.
If you are using python 2, you can currently only read in pickles
that were made in python 2.
If you get a pickled tempate from me (Ben), it will have been made
in python 2 for maxinum compatibility, but yeah, you should be
moving to python 3.

- List of spectra -- text file that must end in '.ls', '.dat', '.txt'. The input file must have the following format: filename order wavelength_range

Some examples:
spec.p 0 *

- (reads in the entire wavelenth range as one order)

spec.p 0 5200-5300,5350-5400
spec.p 0 5400-5600

- (splits the single wavelength array in to two "orders" that are each 200 A wide. The first has a 50 A gap.)

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

If the 'norm' parameter is 'True', the parameter set the width of the
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

- If 'spl', unused regions will be interpolated over with a cubic spline. You probably don't want to use this one.

dk_wav : str

Dictionary keyword for the wavelength array. Default is 'wav'

dk_flux : str

Dictionary keyword for the flux array. Default is 'flux'

*******
Returns
*******
tar : array-like

List of dictionary keywords, described above.

tar_spec : dictionary

SAPHIRES dictionary, described above.


==============================================================================================================
read_fits(spectra_list,temp=False,w_mult=1.0,combine_all=True,norm=True,norm_w_width=200.0,trim_style='clip'):
==============================================================================================================
A function to read in a target spectrum, list of target spectra, or a
template spectrum from an IRAF friendly fits file with a single order,
and put them into the SAPHIRES data structure.

IRAF friendly in this context means that the fits file contrains a flux
array and the headers have keywords CRVAL1, CDELT1, and potentially, LTV1,
that define the wavelength array. If you don't have these keywords, it will
not work. This also assumes the wavelength array has linear spacing.

The "data structure" is nothing special, just a set of nested dictionaries,
but it is a simple way to keep all of the relevant information in one place
that is, hopefully, straight forward.

The returned data structure has two parts:
1) A list of dictionary keywords. They have the following form: "file_name[order][wavelength range]"
2) A dictionary. For each of the keywords above there is a nested dictionary with the following form:
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

**********
Parameters
**********
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

- (splits the single wavelength array in to two "orders" that are each 200 A wide. The first has a 50 A gap.)

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

- If 'spl', unused regions will be interpolated over with a cubic spline. You probably don't want to use this one.

*******
Returns
*******
tar : array-like

List of dictionary keywords, described above.

tar_spec : dictionary

SAPHIRES dictionary, described above.


==============================================================================================================================
read_ms(spectra_list,temp=False,w_mult=1.0,combine_all=True,norm=True,norm_w_width=200.0,trim_style='clip',header_wave=False):
==============================================================================================================================
A function to read in a target spectrum, list of target spectra, or a
template spectrum from an IRAF friendly multi-extension fits file, and
put them into the SAPHIRES data structure.

IRAF friendly in this context means that the fits file contrains a flux
arrays and the headers have 'WAT2' keywords that define the wavelength array.
Currently this only works for linearly spaced wavelength solutions.
Assumes the flux array is in the firsst fits extension, i.e. hdu[0].data.

This is also the function to read in data from the IGRINS spectrograph,
even though those use a separate extension for the wavelength array. Use the
following parameters:
w_mult = 10**4, header_wave=False

The "data structure" is nothing special, just a set of nested dictionaries,
but it is a simple way to keep all of the relevant information in one place
that is, hopefully, straight forward.

The returned data structure has two parts:
1) A list of dictionary keywords. They have the following form: "file_name[order][wavelength range]"
2) A dictionary. For each of the keywords above there is a nested dictionary with the following form:
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

**********
Parameters
**********
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

- (splits the single wavelength array in to two "orders" that are each 200 A wide. The first has a 50 A gap.)

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

- If 'spl', unused regions will be interpolated over with a cubic spline. You probably don't want to use this one.

header_wave : bool or 'Single'

Whether to assign the wavelength array from the header keywords or
from a separate fits extension. If True, it uses the header keywords,
assumiing they are linearly spaced. If False, it looks in the second
fits extension, i.e. hdu[1].data
If header_wave is set to 'Single', it treats each fits extension like
single order fits file that could be read in with saph.io.read_fits.
This feature is useful for SALT/HRS specrtra reduced with the MIDAS
pipeline.

*******
Returns
*******
tar : array-like

List of dictionary keywords, described above.

tar_spec : dictionary

SAPHIRES dictionary, described above.


======================================================================================================================
read_vars(w,f,name,w_file=None,temp=False,combine_all=True,norm=True,w_mult=1.0,trim_style='clip',norm_w_width=200.0):
======================================================================================================================
A function to read in a target spectrum or template spectrum from
predefined python arrays and put them into the SAPHIRES data structure.

Arrays can be single or multi-order. Wavelength and flux arrays have to
have the same dimensionallity and length.

The "data structure" is nothing special, just a set of nested dictionaries,
but it is a simple way to keep all of the relevant information in one place
that is, hopefully, straight forward.

The returned data structure has two parts:
1) A list of dictionary keywords. They have the following form: "file_name[order][wavelength range]"
2) A dictionary. For each of the keywords above there is a nested dictionary with the following form:
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
  ['vflux'] 		- resampled flux array (inverted)

  ['vwave'] 		- resampled wavelength array

  ['vflux_temp'] 	- resampled template flux array (inverted)

  ['vel'] 		- velocity array to be used with the BF or CCF

  ['temp_name'] 	- template name

  ['bf'] 			- broadening funciton

  ['bf_smooth']	- smoothed broadening function

  ['bf_fits'] 	- gaussian fit parameters

More details on these are provided in other various functions.

Many of the spectral analysis techniques (cross correlation, broadening
function, etc) work with a inverted spectrum, i.e. the continuum is at
zero and absorption lines extend upward - so all of the flux arrays are
inverted in this way.

**********
Parameters
**********
w : array-like

Wavelength array, assumed be in Angstroms. If not use the w_mult
keyword below.

f : array-like

Flux array.

name : str

The name of the spectrum.

w_file : str, None

Name of a text file contain wavelength regions that correspond to the
input arrays. The file must have the following format:
order wavelength_range
Some examples:
0 *

- (single order array; reads in the entire wavelenth range as one order)

0 5200-5300,5350-5400
0 5400-5600

- (single order array; splits the single wavelength array in to two "orders" that are each 200 A wide. The first has a 50 A gap.)

0 *
1 5200-5300,5350-5400

- (multi-order array; reads in the entiresy of the first order and a portion of the second order.)

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

- If 'spl', unused regions will be interpolated over with a cubic spline. You probably don't want to use this one.

*******
Returns
*******
tar : array-like

List of dictionary keywords, described above.

tar_spec : dictionary

SAPHIRES dictionary, described above.


================================================
save(t_f_names,t_spectra,outname,text_out=True):
================================================
A function to save a SAPHIRES dictionary as a python pickle file.
The pickle will be saved in whatever python version if currently
being used, i.e. 2 or 3.

**********
Parameters
**********
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

*******
Returns
*******
None
