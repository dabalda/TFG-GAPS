clearvars, clc
addpath(genpath('../'))


paramRW = struct('gamma',.9,'prob_desired_right',.8,'prob_desired_left',.7,'all_states_initial',true);
paramCl = struct('gamma',.9,'slope_east',.0,'slope_north',.0);

% Slow with no preallocation
n_prob = 1000;

tic
problemsRW = [];
for i = 1:n_prob
    problem = RandomWalk(paramRW);
    problemsRW = [problemsRW, problem];
end
toc

tic
problemsCl = [];
for i = 1:n_prob
    problem = Cliff(paramCl);
    problemsCl = [problemsCl, problem];
end
toc
%% Faster with preallocation

clearvars, clc
addpath(genpath('../'))

tic
n_prob = 100;
paramRW = struct('gamma',.9,'prob_desired_right',.8,'prob_desired_left',.7,'all_states_initial',true);
paramCl = struct('gamma',.9,'slope_east',.0,'slope_north',.0);

problemsRW2(1,n_prob) = RandomWalk();
for i = 1:n_prob
    problemsRW2(i) = RandomWalk(paramRW);
end
toc

tic
problemsCl2(1,n_prob) = Cliff();
for i = 1:n_prob
    problemsCl2(i) = Cliff(paramCl);
end
toc

% Test diffVIq

verbose = true;
epsilonPE = 1e-6;

[ PIr, Qr, vr, n_itr ] = diffVIq( problemsRW2, epsilonPE, verbose )
[ PIc, Qc, vc, n_itc ] = diffVIq( problemsCl2, epsilonPE, verbose )