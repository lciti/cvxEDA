"""
______________________________________________________________________________

 File:                         cvxEDA.py
 Last revised:                 07 Nov 2015 r69
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

from typing import Any, Callable
from numpy import array, arange, ndarray, tile, r_, c_, convolve, ones
from cvxopt import matrix, spmatrix, sparse, solvers


def cvxEDA(
    y: ndarray,
    delta: int | float,
    tau0: float = 2.0,
    tau1: float = 0.7,
    delta_knot: float = 10.0,
    alpha: float = 8e-4,
    gamma: float = 1e-2,
    solver: Callable | None = None,
    options: dict[str, Any] = {"reltol": 1e-9},
) -> dict[str, ndarray]:
    # TODO: figure out the actual output format of this method
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
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters

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
    y = matrix(y)

    # bateman ARMA model
    a1 = 1.0 / min(tau1, tau0)  # a1 > a0
    a0 = 1.0 / max(tau1, tau0)
    ar = array(
        [
            (a1 * delta + 2.0) * (a0 * delta + 2.0),
            2.0 * a1 * a0 * delta**2 - 8.0,
            (a1 * delta - 2.0) * (a0 * delta - 2.0),
        ]
    ) / ((a1 - a0) * delta**2)
    ma = array([1.0, 2.0, 1.0])

    # matrices for ARMA model
    i = arange(2, n)
    A = spmatrix(tile(ar, (n - 2, 1)), c_[i, i, i], c_[i, i - 1, i - 2], (n, n))
    M = spmatrix(tile(ma, (n - 2, 1)), c_[i, i, i], c_[i, i - 1, i - 2], (n, n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = r_[arange(1.0, delta_knot_s), arange(delta_knot_s, 0.0, -1.0)]  # order 1
    spl = convolve(spl, spl, "full")
    spl /= max(spl)
    # matrix of spline regressors
    i = (
        c_[arange(-(len(spl) // 2), (len(spl) + 1) // 2)]
        + r_[arange(0, n, delta_knot_s)]
    )
    nB = i.shape[1]
    j = tile(arange(nB), (len(spl), 1))
    p = tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = matrix(c_[ones(n), arange(1.0, n + 1.0) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = solvers.options.copy()
    solvers.options.clear()
    solvers.options.update(options)
    if solver == "conelp":
        # Use conelp
        z = lambda m, n: spmatrix([], [], [], (m, n))
        G = sparse(
            [
                [-A, z(2, n), M, z(nB + 2, n)],
                [z(n + 2, nC), C, z(nB + 2, nC)],
                [z(n, 1), -1, 1, z(n + nB + 2, 1)],
                [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
                [z(n + 2, nB), B, z(2, nB), spmatrix(1.0, range(nB), range(nB))],
            ]
        )
        h = matrix([z(n, 1), 0.5, 0.5, y, 0.5, 0.5, z(nB, 1)])
        c = matrix([(matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = solvers.conelp(c, G, h, dims={"l": n, "q": [n + 2, nB + 2], "s": []})
        obj = res["primal objective"]
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = sparse(
            [
                [Mt * M, Ct * M, Bt * M],
                [Mt * C, Ct * C, Bt * C],
                [
                    Mt * B,
                    Ct * B,
                    Bt * B + gamma * spmatrix(1.0, range(nB), range(nB)),
                ],
            ]
        )
        f = matrix([(matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = solvers.qp(
            H,
            f,
            spmatrix(-A.V, A.I, A.J, (n, len(f))),
            matrix(0.0, (n, 1)),
            solver=solver,
        )
        obj = res["primal objective"] + 0.5 * (y.T * y)
    solvers.options.clear()
    solvers.options.update(old_options)

    l = res["x"][-nB:]
    d = res["x"][n : n + nC]
    t = B * l + C * d
    q = res["x"][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return {
        "phasic component": array(r).ravel(),
        "tonic component": array(t).ravel(),
        "tonic spline coefficients": array(l).ravel(),
        "linear drift term offet and slope": array(d).ravel(),
        "model residuals": array(e).ravel(),
        "objective function value": array(obj).ravel(),
    }
