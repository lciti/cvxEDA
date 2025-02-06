"""
______________________________________________________________________________

 File:                         cvxEDA.py
 Revisions:
    - 07 Nov 2015 First public release
    - 07 Feb 2017 Fixed default alpha to same as paper (8e-4)
    - 22 Jan 2025 Improved boundary behaviour and enhanced knot specification
 ______________________________________________________________________________

 Copyright (C) 2014-2015 Luca Citi, Alberto Greco

 This program is free software; you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation; either version 3 of the License, or (at your option) any later
 version.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

 You may contact the author by e-mail (lciti@ieee.org).
 ______________________________________________________________________________

 This method was first proposed in:
 A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
 "cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
 IEEE Transactions on Biomedical Engineering, 2015
 DOI: 10.1109/TBME.2015.2474131

 If you use this program in support of published research, please include a
 citation of the reference above. If you use this code in a software package,
 please explicitly inform the end users of this copyright notice and ask them
 to cite the reference above in their published research.
 ______________________________________________________________________________
"""

import numpy as np
import cvxopt as cv
import cvxopt.solvers

def cvxEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol': 1e-9}, baseline_correction=2):
    """CVXEDA Convex optimization approach to electrodermal activity processing

    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).

    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: specifies the knots of the tonic spline function; can be a
                   single value (in seconds) representing the spacing between
                   knots, or an array of equally spaced knot sample indices
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
       baseline_correction: baseline correction: 0=none, 1=constant, 2=linear,
                'spline_offset'=non-penalised spline offset

    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """

    n = len(y)
    rangen = np.arange(n)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1. / min(tau1, tau0)  # a1 > a0
    a0 = 1. / max(tau1, tau0)
    ar = np.array([(a1 * delta + 2.) * (a0 * delta + 2.), 2. * a1 * a0 * delta**2 - 8.,
                   (a1 * delta - 2.) * (a0 * delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.concatenate((rangen, rangen[1:], rangen[2:]))
    j = np.concatenate((rangen, rangen[:-1], rangen[:-2]))
    A = cv.spmatrix(np.repeat(ar, (n, n - 1, n - 2)), i, j, (n, n))
    M = cv.spmatrix(np.repeat(ma, (n, n - 1, n - 2)), i, j, (n, n))

    # spline
    if np.size(delta_knot) == 1:  # standard usage: delta_knot represents the interval in seconds between knots
        delta_knot_s = int(round(delta_knot / delta))
        knots = np.arange(0, n + delta_knot_s // 2, delta_knot_s)
    else:  # advanced usage: delta_knot represents an array with indices of the spline knots
        knots = np.reshape(delta_knot, -1)
        delta_knot_s = knots[1] - knots[0]
    spl = np.concatenate((np.arange(1., delta_knot_s), np.arange(delta_knot_s, 0., -1.)))  # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)[:, None] + knots[None, :]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # baseline correction (0=none, 1=constant, 2=linear, 'spline_offset'=non-penalised spline offset)
    if baseline_correction == 'spline_offset':
        C = B * cv.matrix(1., (nB, 1))
        nC = 1
    else:
        nC = baseline_correction
        C = cv.matrix(1., (n, nC)) if nC < 2 else cv.matrix(np.column_stack((np.ones(n), np.arange(1., n + 1.) / n)))

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*q + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m, n: cv.spmatrix([], [], [], (m, n))
        G = cv.sparse([[-A, z(2, n), M, z(nB + 2, n)],
                       [z(n + 2, nC), C, z(nB + 2, nC)],
                       [z(n, 1), -1, 1, z(n + nB + 2, 1)],
                       [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
		               [z(n + 2, nB), B, z(2, nB), cv.spmatrix(1., range(nB), range(nB))]])
        h = cv.matrix([z(n, 1), .5, .5, y, .5, .5, z(nB, 1)])
        c = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cv.solvers.conelp(c, G, h, dims={'l': n, 'q': [n + 2, nB + 2], 's': []})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt * M, Ct * M, Bt * M],
                       [Mt * C, Ct * C, Bt * C],
                       [Mt * B, Ct * B, Bt * B + cv.spmatrix(gamma, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n, len(f))), cv.matrix(0., (n, 1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n : n + nC]
    t = B * l + C * d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))
