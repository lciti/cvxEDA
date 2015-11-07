______________________________________________________________________________

============================= c v x E D A ====================================
______________________________________________________________________________

This program implements the cvxEDA algorithm for the analysis of electrodermal
activity (EDA) using methods of convex optimization, described in:

A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
"cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
IEEE Transactions on Biomedical Engineering, 2015
DOI: 10.1109/TBME.2015.2474131

It is based on a model which describes EDA as the sum of three terms: the
phasic component, the tonic component, and an additive white Gaussian noise
term incorporating model prediction errors as well as measurement errors and
artifacts.
This model is physiologically inspired and fully explains EDA through a
rigorous methodology based on Bayesian statistics, mathematical convex
optimization and sparsity.

The algorithm was evaluated in three different experimental sessions
(see paper) to test its robustness to noise, its ability to separate and
identify stimulus inputs, and its capability of properly describing the
activity of the autonomic nervous system in response to strong affective
stimulation.
______________________________________________________________________________

Copyright (C) 2014-2015 Luca Citi, Alberto Greco

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

If you use this program in support of published research, please include a
citation of the reference above. If you use this code in a software package,
please explicitly inform the end users of this copyright notice and ask them
to cite the reference above in their published research.
______________________________________________________________________________

To use the software from Matlab or Octave, simply call the cvxEDA function in
the src/ folder. Type 'help cvxEDA' from the Matlab/Octave command window for
help on the command's syntax and input/output arguments.

The software does not come with a GUI. Assuming 'y' is a vector with the
recorded EDA signal sampled at 25 Hz, the following example performs the cvxEDA
analysis (with default parameters) and plots the results:

yn = zscore(y);
Fs = 25;
[r, p, t, l, d, e, obj] = cvxEDA(yn, 1/Fs);
figure, hold all
tm = (1:length(y))'/Fs;
plot(tm, yn)
plot(tm, r)
plot(tm, p)
plot(tm, t)
______________________________________________________________________________

To use the software with Python, simply import the cvxEDA module located in
the src/ folder, then call the cvxEDA function. Type 'help(cvxEDA)' from the
python shell for help on the function's syntax and input/output arguments.

The software does not come with a GUI. Assuming 'y' is a numpy vector with the
recorded EDA signal sampled at 25 Hz, the following example performs the cvxEDA
analysis (with default parameters) and plots the results:

import cvxEDA
yn = (y - y.mean()) / y.std()
Fs = 25.
[r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1./Fs)
import pylab as pl
tm = pl.arange(1., len(y)+1.) / Fs
pl.hold(True)
pl.plot(tm, yn)
pl.plot(tm, r)
pl.plot(tm, p)
pl.plot(tm, t)
pl.show()
______________________________________________________________________________
