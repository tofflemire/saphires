.. _example-label:

Worked Examples
***************

Determining the RVs of an SB2
=============================

Here is a quick example using a single-order echelle spectrum taken with the Hydra spectrograph on the WIYN 3.5m telescope. The target is a binary sub-subgiant (a weird star that falls below its host cluster's subgaint branch). For this example I am using a solar spectrum as my template, which could probably be improved by choosing something closer to the T\ :sub:`eff`\ and log(g) of a sub-subgiant.

(These data are provided in the examples directory of the saphires github repo.)

I have provided this example to highlight the broadening function's (BF) performance in a non-ideal case. There is only one order and the spectrum is low S/N. Typically, you will have a full echelle spectrum with many orders that can be combined to produce a high-S/N BF even with weak signal. An example of such a case will be provided soon. For now...


.. code-block:: python

	$ ipython

	In [1]: import saphires as saph

Let's check out the spectrum with the utils.region_select_ms tool to see what we're working with, and select a region over which to compute the BF.

.. code-block:: python

	In [2]: saph.utils.region_select_ms('66212_spec4.fits',header_wave='Single',tar_stretch=False)

It is clear that this is a relatively low S/N spectrum.

In the animation below, I select the region that is not contaminated by edge effects by hitting 'b' to open the window (solid line) and 'b' again to close it (dashed line). Hitting return in the terminal prints the line of text below, which I copy into a file called, 66212_spec4.ls (included in the examples directory).

"66212_spec4.fits 0 4991.84-5243.28"

I have done the same for our template, redtemplate.fits, and have an associated file redtemplate.ls.

.. image:: /figs/highlight_region.gif
	:align: center

Now let's read in the spectrum and template.

.. code-block:: python

	In [3]: tar,tar_spec = saph.io.read_fits('66212_spec4.ls',combine_all=False)

	In [4]: temp = saph.io.read_fits('redtemplate.ls',temp=True)

The next three commands resample the spectra to the same logarithmically spaced wavelength grid (utils.prepare; the same step is required when using a CCF or TODCOR), compute the BF (bf.compute), and simultaneously smooth and fit the BF (bf.analysis). This spectrograph has a resolution of R~18000, but I have lowered the corresponding resolution to R~8000 to reduce the noise in this low S/N spectrum. I have set sb to 'sb2' because I know it is an SB2 -- you'll want to take a look at the BF with an initial guess at the smoothing (the instrumental resolution is a good place to start) and then change the R and sb parameters from there. Setting single_plot = True outputs a pdf with the figures below.

.. code-block:: python

	In [5]: tar_spec = saph.utils.prepare(tar,tar_spec,temp)

	In [6]: tar_spec = saph.bf.compute(tar,tar_spec,vel_width=700)

	In [7]: tar_spec = saph.bf.analysis(tar,tar_spec,R=8000,sb='sb2',single_plot=True)

The first is the wavelength region for the template and spectrum which you have used to compute the BF. The spectra are inverted here by convention.

.. image:: /figs/bf_fig_2.jpg
	:align: center

The second is the smoothed BF and the two fit components.

.. image:: /figs/bf_fig_1.jpg
	:align: center

To return their RVs type:

.. code-block:: python

	In [8]: tar_spec[tar[0]]['bf_rv']
	Out[8]: array([ -11.0376132 , -106.44950092])

The full fit parameters are stored in the tar_spec[tar[0]]['bf_fits'] array, which can be used to determine the flux ratio of the two stars (division of the BF fit integrals).

For comparison, the following plot compared the BF above with a CCF computed for the same region. The BF provides a much cleaner separation between the two components.

.. image:: /figs/bf_ccf.jpeg
	:align: center

Parallelizing BF calculations
=============================


Download the salt fits files and ls files from the repo (in the examples folder)
Create a new directory and move the downloaded files into it.

The fits files are solar spectra taken by the SALT (South Africa Large Telescope). The files contain a total of 55 orders (wavelength sections) that will each be compared against the template spectra (lte055..... .p)

Our goal here is to establish the radial velocity of the star observed in the SALT spectra. We can do this by calculating a broadening function with each order against the template. You could average out the 55 radial velocity results to find a good estimated value, but in this example we just plot them all together in the same pdf.

after pip installing saphires (if you dont have it) open a terminal and cd into the new directory with the SALT files
Now open your python REPL
Mac OSX
.. code-block:: python
   $ python3
Windows
.. code-block:: python
   $ python

Now we will import saphires so we can use its functionality
.. code-block:: python
	import saphires as saph

Here we are using the ls file (that splits up the wavelengths of the SALT fits into sub sections) to create a saphires spectra dictionary
.. code-block:: python
	tar,tar_spec = saph.io.read_ms('./salt1_rb.ls',combine_all=False,header_wave='Single')

Next we read in the template from the pikl file
.. code-block:: python
	temp = saph.io.read_pkl('./lte05500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_2800-11000_air.p',temp=True)

No we prepare the spectra with the template spectra
.. code-block:: python
	tar_spec = saph.utils.prepare(tar,tar_spec,temp)

Here is where we do the broadening function calculations. Currently multiple_p is set to True, meaning the calculations will be done in a parallel fashion. This is going to speed up the calculation process(from 6 minutes to 4.5 minutes on my computer). If you are ok with the slower version, you can set multiple_p to False. No multiple processing will also prevent much lag on your machine while running the calculations.
.. code-block:: python
	tar_spec = saph.bf.compute(tar,tar_spec,vel_width=400,multiple_p = True)

This is where we plot all the BF graphs, it should create a 110 page pdf in the same directory as the SALT files.
.. code-block:: python
	tar_spec = saph.bf.analysis(tar,tar_spec,R=50000,single_plot=True)
