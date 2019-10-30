Introduction
************
SAPHIRES -- Stellar Analysis in Python for HIgh REsolution Spectroscopy -- is a python module that hosts a suite of functions for the analysis of high-resolution stellar spectra. Most of the functionality is aimed at deriving radial velocities (RVs), but includes capabilities to measure projected rotational velocities (vsini) and determine spectroscopic flux ratios in double-lined binary systems (SB2s). These measurements are made primarily by computing spectral-line broadening functions. More traditional techniques such as Fourier cross-correlation, and two-dimensional cross-correlation (TODCOR) are also included. The sections below provide an overview of each technique.

Spectral-Line Broadening Functions (BF)
=======================================
Spectral-line broadening functions result from a linear inversion of an observed spectrum with a zero-velocity (ideally, narrow-lined) template. This implementation performs the inversion with singular value decomposition following Rucinski (1999) [#f1]_ [#f2]_. The raw broadening function represents the function that would need to be convolved with the template in order to reproduce the observed spectrum. For instance, a simple radial-velocity shift with an otherwise perfectly matched template spectrum would result in a delta function at the shift velocity. 

In practice, the raw broadening function is dominated by noise from the observed spectrum (and template spectrum if using an empirical template), and from template mismatch. Additionally, elements of the raw broadening function are not independent on scales smaller than the instrumental velocity resolution. When convolved with a Gaussian corresponding to the template's line width, the smoothed broadening function (hereafter BF) has the advantageous property of representing the average photospheric line profile of the observed spectrum. This makes the BF ideal for measuring the RV and vsini of a source, and detecting velocity-resolved surface brightness features (i.e. Doppler tomography). In the case of a spectroscopic binary system (SB2), the relative contribution of each star (flux ratio) can be measured as the ratio of the integral of each BF component. 

While the BF is similar to a cross-correlation function (CCF), it has features that make it superior for determining RVs and vsini values in many scenarios. The main advantage is that the BF represents the actual photospheric absorption line profile, rather than a Fourier abstraction of it. The BF is flat (and zero) on either side of the reconstructed line, making it possible to fit the line with a rotationally broadened profile, for instance, to directly measure the vsini and RV. The CCF for rotationally broadened stars can become difficult to interpret as the peak blends into fringing sidebands. Measuring vsini with a CCF usually entails creating a grid of templates with different levels of rotational broadening, computing a CCF for each template to see which produces the highest CCF height. With a BF, you get all this for free, and can even assess the level of microturbulent broadening if you are feeling brave. I also generally find that BFs perform better than CCFs and TODOCR at low signal-to-noise.

In the case of SB2s, BF are superior to CCFs in every way. CCFs suffer from "peak-pulling" in SB2s, which underestimate the orbit's RV amplitude. TODCOR is fine to use for SB2s, and is actually better than a BF for small velocity separations (see below), but do not use the CCF implementation provided here or anywhere else for a SB2! There is a great paper by Cunningham et al. (2019) [#fa]_, which displays this very nicely, also see the :ref:`example-label` page.

For a high-level walkthrough on how BFs are computed, check out the :ref:`bfwt-label` page.

You do not get all this stuff for free:

* BFs are more computationally expensive than CCFs. 
* Both are sensitive the template mismatch, but BFs are particularly sensitive to emission features (astrophysical or cosmetic), which need to be masked to recover a clean BF. 
* In the case of SB2s, the BF is limited in that it applies a single template to stars that may not have identical spectra. This scenario is explored in Tofflemire et al. (2019) [#f3]_, finding that RV measurements are generally robust to differences in stellar temperature, while the flux ratio is more sensitive to the template used, but is readily recovered by the template best matching the primary star.
* As a reconstruction of the stellar line profile, SB2 epochs at small velocity separations can be difficult to decompose. When determining a orbital solution, I usually use BF where I can and TODCOR for overlapping epochs.

These drawbacks aside, I typically find that BF out preforms the methods below, are more straight forward to work with, and more forgiving when dealing with binary and higher-order systems (SB3s, SB4s). BFs will perform best when you have high spectral resolution and large wavelength coverage. 


Two-Dimensional Cross Correlation (TODCOR)
==========================================

The TODCOR algorithm was developed by Zucker & Mazeh (1994) [#f4]_, and is implemented here. TODCOR constructs a two-dimensional surface where x-y values represent the RV of the primary and secondary, where the z value is the Fourier cross correlation between the observed spectrum and a template constructed from two templates at the corresponding RVs. The highest peak in the TODCOR surface corresponds to the primary and secondary RVs of your source. This algorithm also provides a measure of the flux ratio for each component. The CCFs are computed following the Tonry and Davis (1979) [#fb]_ definition where results is normalized by the number of overlapping spectral bins, rather than the total array length. 

The main limitation of TODCOR (similar to BFs and CCFs) is that the results are only as good as the input templates. TODCOR is more flexible than a BF or CCF in that you have the ability to use templates that are a better match to the individual stars in a binary, but I find that the results are more sensitive mismatches, especially the flux ratio measurements. If you have full knowledge of the system (i.e., T\ :sub:`eff, 1`\, T\ :sub:`eff, 2`\, vsini\ :sub:`1`\, vsini\ :sub:`2`\) TODCOR will do great, but without this knowledge, BFs are much more forgiving. Both BFs and TODCOR are very complementary and I often use both in concert. 


Fourier Cross Correlation Function (CCF)
========================================

The normalized Fourier power as a function of the template's RV. CCFs are computed following the Tonry and Davis (1979) [#fb]_ definition where results is normalized by the number of overlapping spectral bins, rather than the total array length. It produces a peak at the RV that provides the most similarity between the template and observed spectrum. The CCF is a useful and computationally simple tool that can easily determine the RV of a single star when the system is well-behaved, has known parameters, and high-signal to noise. This is the main function used in the HARPS pipeline for instance, which does an excellent job measuring planet masses. The success of their version relies on accurate weights for different spectral orders and the use of a binary mask as the template. For cases where these conditions are not met, other algorithms provide more reliable results. 

(There is currently not an easily accessible CCF in SAPHIRES. There is a utils function that is called by todcor.)


Utility Functions
=================

SAPHIRES also has a lot of useful utility functions:

* Air to vacuum wavelength conversions (and vice versa)
* Bspline continuum normalization
* Interactive region selection tools for masking bad wavelengths
* Rotationally broadened line profiles for fitting BFs or for convolving with spectra
* And more!



.. [#f1] `Rucinski (1999) <https://ui.adsabs.harvard.edu/abs/1999TJPh...23..271R/abstract>`_
.. [#f2] `Rucinski's IDL implementation <http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html>`_
.. [#fa] `Cunningham et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019AJ....158..106C/abstract>`_
.. [#f3] `Tofflemire, Mathieu, and Johns-Krull (2019) <https://arxiv.org/abs/1910.12878>`_
.. [#fb] `Tonry and Davis (1979) <https://ui.adsabs.harvard.edu/abs/1979AJ.....84.1511T/abstract>`_
.. [#f4] `Zucker & Mazeh 1994 <https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract>`_
	