# cvxEDA

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

The algorithm was evaluated in three different experimental sessions
(see paper) to test its robustness to noise, its ability to separate and
identify stimulus inputs, and its capability of properly describing the
activity of the autonomic nervous system in response to strong affective
stimulation.

## Repository layout

| Directory | Language | Description |
|-----------|----------|-------------|
| `python/` | Python   | `cvxeda` Python package |
| `matlab/` | MATLAB / Octave | `cvxEDA.m` and test scripts |
| `README_PYTHON.md` | – | Python‑specific installation and usage |
| `README_MATLAB.md` | – | MATLAB/Octave‑specific usage |
| `LICENSE.txt` | – | GPL‑3.0 license text. |

## Quick links

* **Python usage:** [`README_PYTHON.md`](README_PYTHON.md)
* **MATLAB/Octave usage:** [`README_MATLAB.md`](README_MATLAB.md)

## License & citation

The code is released under **GPL‑3.0** (see [`LICENSE.txt`](LICENSE.txt)).

If you use this program in support of published research, **please cite the
reference above**. If you use this code in a software package, please
explicitly inform the end users of this copyright notice and ask them
to cite the reference above in their published research.

*Project repository:* <https://github.com/lciti/cvxeda>
