function [ref_pair, test_pair] = bxb_simple(ref, test, match_dt)
%BXB_SIMPLE simple matlab implementation of bxb.c from wfdb
%   (see http://www.physionet.org/physiotools/wfdb/app/bxb.c).
%   The algorithm is modelled after the AAMI/ANSI EC38:1998 standard.
%   Briefly, times in 'ref' and times in 'test' are considered as matching
%   if they are within a match window of duration 'match_dt'.
%   Each event from either input can only match a single event from the other
%   one.
%
%   Syntax:
%   [ref_pair, test_pair] = bxb_simple(ref, test, match_dt)
%
%   where:
%      ref: is the set of times of the reference events
%      test: is the set of times of the test events
%      match_dt: tolerance (in the same unit as the times in ref and test)
%
%   returns:
%      ref_pair: index of test events matching each ref event (0 if unmatched)
%      test_pair: index of ref events matching each test event (0 if unmatched)

% ______________________________________________________________________________
%
% File:                         bxb_simple.m
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
% The original c code was written by George B Moody:
% GB Moody
% "bxb: ANSI/AAMI-standard beat-by-beat annotation file comparator"
% Online at http://www.physionet.org/physiotools/wfdb/app/bxb.c
% Retrieved 11 Apr 2015
%
% This implementation was used in:
% A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
% "cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
% IEEE Transactions on Biomedical Engineering, 2015
% DOI: 10.1109/TBME.2015.2474131
%
% If you use this program in support of published research, please include a
% citation of the references above. If you use this code in a software package,
% please explicitly inform the end users of this copyright notice and ask them
% to cite the references above in their published research.
% ______________________________________________________________________________

ref_pair = zeros(size(ref));
test_pair = zeros(size(test));

ref(end+1) = inf;
test(end+1) = inf;

i_r = 1;
i_t = 1;

while (i_r < length(ref) && i_t < length(test))

    % fetch times
    T = ref(i_r);
    Tprime = ref(i_r+1);
    t = test(i_t);
    tprime = test(i_t+1);

    if (t < T) % test annotation is earliest

        % (1) If t is within the match window, and is a better match than
        %     the next test annotation, pair it.
        if (T-t <= match_dt && ...
               (T-t < abs(T-tprime) || abs(Tprime-tprime) < abs(T-tprime)))
            % pair ref and test
            ref_pair(i_r) = i_t;
            test_pair(i_t) = i_r;
            % move to next ref and test
            i_t = i_t + 1;
            i_r = i_r + 1;

        % (2) There is no match to the test annotation, so pair it
        %     with a pseudo-beat annotation and get the next one.
        else
            % pair test with pseudo-beat
            test_pair(i_t) = 0;
            % move to next test
            i_t = i_t + 1;
        end

    else % reference annotation is earliest

        % (3) If T is within the match window, and is a better match than
        % the next reference annotation, pair it.
        if (t-T <= match_dt && ...
               (t-T < abs(t-Tprime) || abs(tprime-Tprime) < abs(t-Tprime)))
            % pair ref and test
            ref_pair(i_r) = i_t;
            test_pair(i_t) = i_r;
            % move to next ref and test
            i_t = i_t + 1;
            i_r = i_r + 1;

        % (4) There is no match to the ref annotation, so pair it
        %     with a pseudo-beat annotation and get the next one.
        else
            % pair ref with pseudo-beat
            ref_pair(i_r) = 0;
            % move to next ref
            i_r = i_r + 1;
        end
    end
end
