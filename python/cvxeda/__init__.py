"""cvxEDA – Python implementation

This program implements the cvxEDA algorithm for the analysis of electrodermal
activity (EDA) using methods of convex optimisation, described in:

A. Greco, G. Valenza, A. Lanata, E. P. Scilingo, and L. Citi  
“cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing”  
IEEE Transactions on Biomedical Engineering, 2015  
https://doi.org/10.1109/TBME.2015.2474131


What the algorithm does
-----------------------

It is based on a model which describes EDA as the sum of three terms:

- phasic component – transient increases reflecting sudomotor bursts;
- tonic component – slowly varying baseline activity;
- additive white‑Gaussian‑noise term – captures model prediction errors,
  measurement noise and artifacts.

The model is physiologically inspired and fully explains EDA through a
rigorous methodology based on Bayesian statistics, convex optimisation and
sparsity.


Quick example
-------------

>>> import numpy as np
>>> import cvxeda
>>> import matplotlib.pyplot as plt
>>> 
>>> # Simulated EDA signal (replace with your own data)
>>> tm = np.linspace(0, 40, 1000)                                           # 40 s at 25 Hz
>>> y = np.convolve(np.sin(.7 * tm) < -.8, np.exp(-.5 * tm[:200]), 'same')  # simple fake signal
>>> y += np.random.randn(len(y))                                            # add noise
>>> 
>>> yn = (y - y.mean()) / y.std()  # standardise
>>> Fs = 25.0                      # sampling frequency (Hz)
>>> 
>>> # Run the algorithm
>>> r, p, t, l, d, e, obj = cvxeda.cvxEDA(yn, 1.0 / Fs)
>>> 
>>> # Plot
>>> tm = np.arange(1, len(y) + 1) / Fs
>>> plt.figure()
>>> plt.plot(tm, yn, label='normalised input')
>>> plt.plot(tm, r,  label='phasic component')
>>> plt.plot(tm, p,  label='sparse SMNA driver')
>>> plt.plot(tm, t,  label='tonic component')
>>> plt.xlabel('Time (s)')
>>> plt.legend()
>>> plt.show()


Help
----

Full function signature and argument description:
>>> help(cvxeda.cvxEDA)


License & citation
------------------

The code is released under GPL‑3.0.

If you use this program in support of published research, please cite the
reference above. If you use this code in a software package, please
explicitly inform the end users of this copyright notice and ask them
to cite the reference above in their published research.

Project repository: <https://github.com/lciti/cvxeda/blob/main/README.md>

"""

from .cvxEDA import cvxEDA
from ._version import __version__, __version_tuple__
