# cvxEDA – MATLAB/Octave implementation

This program implements the **cvxEDA** algorithm for the analysis of
electrodermal activity (EDA) using methods of convex optimisation, described
in:

A. Greco, G. Valenza, A. Lanata, E. P. Scilingo, and L. Citi  
**“cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing”**  
*IEEE Transactions on Biomedical Engineering*, 2015  
DOI: [10.1109/TBME.2015.2474131](https://doi.org/10.1109/TBME.2015.2474131)

## What the algorithm does

It is based on a model which describes EDA as the sum of three terms:

* **phasic component** – transient increases reflecting sudomotor bursts;
* **tonic component** – slowly varying baseline activity;
* **additive white‑Gaussian‑noise term** – captures model prediction errors,
  measurement noise and artifacts.

The model is physiologically inspired and fully explains EDA through a
rigorous methodology based on Bayesian statistics, convex optimisation and
sparsity.

## Quick example

```matlab
% Simulated EDA signal (replace with your own data)
tm = linspace(0, 40, 1000);                                  % 40 s at 25 Hz
y = conv(sin(.7 * tm) < -.8, exp(-.5 * tm(1:200)), 'same');  % simple fake signal
y = y + randn(size(y));                                      % add noise

yn = zscore(y);            % standardise
Fs = 25;                   % sampling frequency (Hz)

% Run the algorithm
[r, p, t, l, d, e, obj] = cvxEDA(yn, 1/Fs);

% Plot the results
tm = (1:length(y))' / Fs;
figure; hold on
plot(tm, yn, 'DisplayName', 'normalised input')
plot(tm, r,  'DisplayName', 'phasic component')
plot(tm, p,  'DisplayName', 'sparse SMNA driver')
plot(tm, t,  'DisplayName', 'tonic component')
xlabel('Time (s)')
legend
```

`help cvxEDA` displays the full function signature and a description of the
inputs/outputs.

## License & citation

The code is released under **GPL‑3.0** (see [`LICENSE.txt`](LICENSE.txt)).

If you use this program in support of published research, **please cite the
reference above**. If you use this code in a software package, please
explicitly inform the end users of this copyright notice and ask them
to cite the reference above in their published research.

*Project repository:* <https://github.com/lciti/cvxeda>
