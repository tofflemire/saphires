from distutils.core import setup

setup(name='saphires',
      version='0.1.0',
      description='Stellar Analysis in Python for HIgh-REsolution Spectroscopy',
      author='Ben Tofflemire',
      author_email='tofflemire@utexas.edu',
      license='BSD/MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Astronomy',
          ],
      packages=['saphires','saphires.extras'],
      requires=['numpy',
                'astropy',
                'scipy',
                'matplotlib',
                'pickle',
                'PyQt5'])
