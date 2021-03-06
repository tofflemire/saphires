========================================================================================
compute(t_f_names,t_spectra,vel_width=200,quiet=False,matrix_out=False,multiple_p=True):
========================================================================================
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

- You probably won't know the right range ahead of time so start broad

- Think about applying a shift to your spectrum with saphires.utils.apply_shit to center the feature(s) you care about, allowing you to compute the BF over a smaller velocity range

- It is good to have an equal amount of velocity space computed within the features you care about and outside of them

**********
Parameters
**********
t_f_names: array-like

Array of keywords for a science spectrum SAPHIRES dictionary. Output of
one of the saphires.io read-in functions.

t_spectra : python dictionary

SAPHIRES dictionary for the science spectrum that has been prepared with
the utils.prepare function with a template spectrum.

vel_width : float

The range over which to compute the broadening function. The input
value is the amount of velocity space conputed on either size of 0,
i.e. a value of 200 produces a BF that spans -200 to +200. Units
are in km/s.  The default value is 200 km/s.
Large values take longer to run. Small values could exclude a peak
you care about.

quiet : bool

Specifies whether messaged are printed to the teminal. Specifically, if
there is not enough velocity space in the input spectra,this function
will print and error. The default value is False.

matrix_out : bool

Specifies whether the "lower order" BFs, those comprised of less
eigenvectors, and their assocaited weights should be returned in the
output dictionary. This can be useful in some cases to help you determine
the level of smoothing to use. It greatly inflated the size of the
dictionaries, and so it is left as a option to the user.
The default is False, to not return this matrix. The matrix is
described in the Returns section below.

multiple_p: bool

Specifies if the computation of broadening functions should be done
using parallel processing. Setting this as True will mean faster computation
but your computer may be a bit laggy. Setting this as False will mean slower
computation times, but your computer will run normally. Default is True

*******
Returns
*******
spectra : python dictionary

A python dictionary with the SAPHIRES architecture. The output dictionary
will have 3 (or 5) new keywords as a result of this function. And is a copy
of t_spectra.

['vel'] - Velocity array over which the BF is computed

['bf'] - The unsmoothed BF array

['bf_sig'] - The sigma on the BF - proxy for error on the fit (a single value)

If matrix_out == True

['bf_matrix'] - A matrix of the lower order BFs: each row is a BF made with an increasing numer of eigenvectors. The last element is provided inn the 'bf' keyword above.

['bf_sig_array'] - The sigma is the associated sig for each BF in the matrix above (array)

It also updates the values for the following keyword under the right
conditions:

['order_flag'] 	- order flag will be updated to 0 if the order has less velocity space than is asked to compute.


=======================================================================================================
weight_combine(t_f_names,spectra,std_perc=0.1,vel_gt_lt=None,bf_sig=False,bf_ind=False,sig_clip=False):
=======================================================================================================
A function to combine BFs from different spectral orders, weighted
by the standard deviation of the BF sideband.

BF can only be combined if you prepared the spectra using the option
vel_spacing="uniform", which is the default.

The STD of their sidebands (as determined with the std_perc or
vel_gt_lt). A three is an optional sigma_clip parameter to remove
huge outliers.

The surviving BFs are combined, weighted by the sideband STD.

**********
Parameters
**********
t_f_names: array-like

Array of keywords for a science spectrum SAPHIRES dictionary. Output of
one of the saphires.io read-in functions.

t_spectra : python dictionary

SAPHIRES dictionary for the science spectrum that has been prepared with
the utils.prepare function with a template spectrum.

std_perc : float

Defines the sideband region to determine each order's weight.
The value is the percentage of the velocity space, over which the entire
BF was computed, to be used to measure the sideband standard deviation,
evaluated from each end. For example, if std_perc = 0.1 (i.e. 10%), and
the BF was computed over +/- 200 km/s (400 km/s total), a 40 km/s region
on either end of the BF will be used to determine the order standard
deviation.
This option is nice when your features are centered near zero velocity.
An alternative options is available with the vel_gt_lt parameter.
The default value if 0.1

vel_gt_lt : array-like

A two element array providing the upper and lower limit of the velocity
array over which the BF standard deviation is computed. For example, if
your feature is at +10 km/s and is 20 km/s wide, you could enter
vel_gt_lt = (+35,-5). If this parameter is used, std_perc is ignored.
The default value is None.

bf_sig : bool

Option to use the formal BF sigma (goodness of fit) as the weighting
factor. The defalt is False

bf_ind : bool, int


sig_clip : bool

Option to perform a sigma clip on the measured standard deviation.
The default value is False (if your weights make sense, you should not
need this step).

*******
Returns
*******
v : array-like

The velocity array of the weighted, combined BF.

bf_wsc : array-like

The weighted, combined BF.

bf_wsc_sterr : float

The standard error derived from the weights. A single value that
applied to all velocity elements of the combined BF array

bf_wsc_ewstd : array-like

The error-weighted standard deviation of the combined BF. An
array of the same length as v and bf_wsc

======================================================================================================================================================
analysis(t_f_names,t_spectra,sb='sb1',fit_trim=20,text_out=False,text_name=False,single_plot=False,p_rv=False,prof='g',R=50000.0,R_ip=50000.0,e=0.75):
======================================================================================================================================================
A function to analyze broadening functions. This will smooth the
BF and attempt to fit a specified number of line profiles to it
in order to measure stellar RVs and determine their flux ratios,
in the case of a spectrscopic binary or triple.

DISCLAIMER - This function is a beast, but it does most of what
I want it to do with the values below with the default values
that are hard coded below. Eventually, there should be keyword
arguments for the fit bounds etc., but for now if you need to
change them, I'm afraid you'll have to alter this code.

**********
Parameters
**********
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

- The Gaussian profile has 4 parameters: amplitude (not normalized), center (this is the RV), standard deviation (sigma), and vertical offset.

- The rotataionally broadened profile has 4 paramteres: amplitude (normalized, i.e., the amplitude is the profile integral), center (RV), width (vsini), and vertical offset. The profile uses a linear limb-darkening law that is specified with the 'e' keyword (limb darkening is not fit). Analytic formula from Gray 2005. The default is 'g', a Gaussian.

R : float

Defines the smoothing element. The first staring place should the be
spectrograph's resolution. Smoothing at higher resolutions doesn't make sense.
Additional smoothing, i.e., a lower resolution may be advantageous for noisey
BFs or very heavily rotationally broadened profiles. The rotatinoally broadened
profile is convolved with this level of smoothing as well. The default is 50000.

e : float

Linear limb darkening parameter. Default is 0.75,
appropriate for a low-mass star.

*******
Returns
*******
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
