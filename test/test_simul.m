%TEST_SIMUL simple script to test reconstruction accuracy of the cvxEDA method
%   (see http://dx.doi.org/10.1109/TBME.2015.2474131).

% ______________________________________________________________________________
%
% File:                         test_simul.m
% Last revised:                 01 Oct 2015 r67
% ______________________________________________________________________________
%
% Copyright (C) 2015 Luca Citi
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
% This algorithm was used in:
% A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
% "cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
% IEEE Transactions on Biomedical Engineering, 2015
% DOI: 10.1109/TBME.2023.24741
%
% If you use this program in support of published research, please include a
% citation of the reference above. If you use this code in a software package,
% please explicitly inform the end users of this copyright notice and ask them
% to cite the reference above in their published research.
% ______________________________________________________________________________

close all
clearvars

addpath('../src/')

% SIMULATION
num_runs = 100;
delta = 1/50;
T = 90;
min_ISI = 1;
num_spk = 10;
tau1 = .7;
tau0_min = 2;
tau0_max = 4;
sin_min = 45;
sin_max = 90;
n_std = [1e-2 1e-1];
threshold = 0.5;
match_dt = .15;

% ESTIMATION
fit_edr_model = @cvxEDA
delta_knot = 10; % sec
alpha = 8e-4;
gamma = 1e-2;
tau0_step = .2;

% LOOP
t = (0:delta:T)';

t0 = tic();
for j = num_runs:-1:1
    
    % smna
    while true
        spk_i = sort(randi(length(t), [num_spk 1]));
        if all(diff([0; t(spk_i); T]) > min_ISI), break, end;
    end

    spk = zeros(size(t));
    spk(spk_i) = 1/delta;

    % phasic
    tau0 = tau0_min + rand() * (tau0_max - tau0_min);
    alpha1 = delta / tau1;
    alpha0 = delta / tau0;
    den = [(1+alpha1)*(1+alpha0), -2-alpha1-alpha0, 1];
    num = delta * (alpha1 - alpha0) / den(1);
    den = den / den(1);
    y_r = filter(num, den, spk);

    % tonic
    y_t = 2 + rand()*linspace(0,1,length(t))' + sin(2*pi*(rand()+t/(sin_min ...
            + rand()*(sin_max - sin_min))));

    for nn = 1:length(n_std)
       
        y_n = y_r + y_t + n_std(nn)*randn(size(t));

        % estimation
        tau0_ = tau0_min:tau0_step:tau0_max;
        for i = length(tau0_):-1:1
            [phasic{i}, smna{i}, tonic{i}, l{i}, d{i}, e{i}] = fit_edr_model(...
                y_n(:), delta, tau1, tau0_(i), delta_knot, alpha, gamma);
        end
        res_e = cellfun(@norm, e);
        [~,best] = min(res_e);
        
        % measure phasic accuracy
        area_pulses = conv(smna{best}, [1 1 1]) * delta;
        test = t(1 + find(diff(area_pulses > threshold) > 0));
        ref = t(spk_i);
        [ref_pair, test_pair] = bxb_simple(ref, test, match_dt);
        SENS(j,nn) = mean(ref_pair > 0);
        PPV(j,nn) = mean(test_pair > 0);
    end

    fprintf('%d of %d (ETA: %s)\n', j, num_runs, ...
            datestr(now()+toc(t0)/86400*(j-1)/(num_runs-j+1)))
end
toc(t0)

figure
plot(tau0_, res_e)

figure, hold all
plot(t, y_n)
plot(t, y_t)
plot(t, tonic{best})
plot(t, spk*delta)
plot(t, smna{best}*delta)
