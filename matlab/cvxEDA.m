function [r, p, t, l, d, e, obj] = cvxEDA(y, delta, varargin)
%CVXEDA Convex optimization approach to electrodermal activity processing
%   This function implements the cvxEDA algorithm described in "cvxEDA: a
%   Convex Optimization Approach to Electrodermal Activity Processing"
%   (http://dx.doi.org/10.1109/TBME.2015.2474131 also available from the
%   authors' homepages).
%
%   Syntax:
%   [r, p, t, l, d, e, obj] = cvxEDA(y, delta, tau0, tau1, delta_knot,
%                                    alpha, gamma, solver)
%
%   where:
%      y: observed EDA signal (we recommend normalizing it: y = zscore(y))
%      delta: sampling interval (in seconds) of y
%      tau0: slow time constant of the Bateman function (default 2.0)
%      tau1: fast time constant of the Bateman function (default 0.7)
%      delta_knot: time between knots of the tonic spline function (default 10)
%      alpha: penalization for the sparse SMNA driver (default 0.0008)
%      gamma: penalization for the tonic spline coefficients (default 0.01)
%      solver: sparse QP solver to be used, 'quadprog' (default) or 'sedumi'
%
%   returns (see paper for details):
%      r: phasic component
%      p: sparse SMNA driver of phasic component
%      t: tonic component
%      l: coefficients of tonic spline
%      d: offset and slope of the linear drift term
%      e: model residuals
%      obj: value of objective function being minimized (eq 15 of paper)

% ______________________________________________________________________________
%
% File:                         cvxEDA.m
% Last revised:                 22 Oct 2015 r68
% ______________________________________________________________________________
%
% Copyright (C) 2014-2015 Luca Citi, Alberto Greco
%
% This program is free software; you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation; either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You may contact the author by e-mail (lciti@ieee.org).
% ______________________________________________________________________________
%
% This method was first proposed in:
% A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
% "cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
% IEEE Transactions on Biomedical Engineering, 2015
% DOI: 10.1109/TBME.2015.2474131
%
% If you use this program in support of published research, please include a
% citation of the reference above. If you use this code in a software package,
% please explicitly inform the end users of this copyright notice and ask them
% to cite the reference above in their published research.
% ______________________________________________________________________________

% parse arguments
params = {2, 0.7, 10, 8e-4, 1e-2, 'quadprog'};
i = ~cellfun(@isempty, varargin);
params(i) = varargin(i);
[tau0, tau1, delta_knot, alpha, gamma, solver] = deal(params{:});

n = length(y);
y = y(:);

% bateman ARMA model
a1 = 1/min(tau1, tau0); % a1 > a0
a0 = 1/max(tau1, tau0);
ar = [(a1*delta + 2) * (a0*delta + 2), 2*a1*a0*delta^2 - 8, ...
       (a1*delta - 2) * (a0*delta - 2)] / ((a1 - a0) * delta^2);
ma = [1 2 1];

% matrices for ARMA model
i = 3:n;
A = sparse([i i i], [i i-1 i-2], repmat(ar, n-2, 1), n, n);
M = sparse([i i i], [i i-1 i-2], repmat(ma, n-2, 1), n, n);

% spline
delta_knot_s = round(delta_knot / delta);
spl = [1:delta_knot_s delta_knot_s-1:-1:1]'; % order 1
spl = conv(spl, spl, 'full');
spl = spl / max(spl);
% matrix of spline regressors
i = bsxfun(@plus, (0:length(spl)-1)'-floor(length(spl)/2), 1:delta_knot_s:n);
nB = size(i, 2);
j = repmat(1:nB, length(spl), 1);
p = repmat(spl(:), 1, nB);
valid = i >= 1 & i <= n;
B = sparse(i(valid), j(valid), p(valid));

% trend
C = [ones(n,1) (1:n)'/n];
nC = size(C, 2);

% Solve the problem:
% .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
% s.t. A*q >= 0

if strcmpi(solver, 'quadprog')
    % Use Matlab's quadprog
    H = [M'*M, M'*C, M'*B; C'*M, C'*C, C'*B; B'*M, B'*C, B'*B+gamma*speye(nB)];
    f = [alpha*sum(A,1)'-M'*y; -(C'*y); -(B'*y)];

    [z, obj] = quadprog(H, f, [-A zeros(n,length(f)-n)], zeros(n, 1), ...
        [], [], [], [], [], optimset('Algorithm', 'interior-point-convex', ...
        'TolFun', 1e-13));
    %z = qp([], H, f, [], [], [], [], zeros(n,1), [A zeros(n,length(f)-n)], []); 
    obj = obj + .5 * (y' * y);
elseif strcmpi(solver, 'sedumi')
    % Use SeDuMi 
    U = [A, sparse(n,nC), -speye(n), sparse(n,n+nB+4); ...
         M, C, sparse(n,n+2), -speye(n), sparse(n,2), B; ...
         sparse(1,2*n+nC), 1, sparse(1,n+nB+3); ...
         sparse(1,3*n+nC+2), 1, sparse(1,nB+1)];
    b = [sparse(n,1); y; 1; 1];
    c = sparse([n+nC+(1:n), 2*n+nC+2, 3*n+nC+4], ...
               1, [alpha*ones(1,n), 1, gamma], 3*n+nC+nB+4, 1);
    K = struct('f', n+nC, 'l', n, 'r', [2+n 2+nB]);
    pars.eps = 1e-6;
    pars.chol.maxuden = 1e2;
    z = sedumi(U, b, c, K, pars);
    obj = c' * z;
    %objd = b' * s;
end

l = z(end-nB+1:end);
d = z(n+1:n+nC);
t = B*l + C*d;
q = z(1:n);
p = A * q;
r = M * q;
e = y - r - t;

end
