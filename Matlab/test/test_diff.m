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
problems_train = [];
for i = 1:n_prob
    problem = Cliff(paramCl);
    problems_train = [problems_train, problem];
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

%% Different problems

clearvars, clc
addpath(genpath('../'))

n_train = 10;

maxN = .1;
minN = -.1;
maxE = .1;
minE = -.1;

rangeN = maxN - minN;
rangeE = maxE - minE;

slope_east = rangeE*rand(n_train,1)+minE;
slope_north = rangeN*rand(n_train,1)+minN;

tic
problems_train(1,n_train) = Cliff();
for i = 1:n_train
    param = struct('gamma',.9,'slope_east',slope_east(i),'slope_north',slope_north(i));
    problems_train(i) = Cliff(param);
end
toc
%%
epsilon = 1e-6;

% Get Diffusion policy
tic
[PI_diff, Q_diff, v_diff, n_it_diff ] = diffVIq(problems_train, epsilon, true);
toc
%%
% Get optimal value for each problem while testing number of policy
% iterations with PIv starting from diffusion policy
v_opt_train = zeros(problems_train(1).n_states, n_train);
v_diff_train = v_opt_train;
n_it_PI_from_diff = zeros(n_train,1);
PI_opt = zeros(problems_train(1).n_states, problems_train(1).n_actions, n_train);

tic
for p = 1:n_train
    [PI_opt(:,:,p) ,v_opt_train(:,p), ~, n_it_PI_from_diff(p)] = PIv(problems_train(p), epsilon, PI_diff);
    v_diff_train(:,p) = PEv(problems_train(p), PI_diff, epsilon);
end
toc

v_opt_mean = mean(v_opt_train,2)
v_diff_mean = mean(v_diff_train,2)
sum_v_opt_mean = sum(v_opt_mean)
sum_v_diff_mean = sum(v_diff_mean)
n_it_PI_from_diff_mean = mean(n_it_PI_from_diff)

%% DiffSARSA

n_episodes = 1000;
epsilonRL = .1;
alpha = .05;
discount_threshold = 0;
tolerance = 1e-3;
verbose = true;

[ PI_dS, Q_dS, v_dS ] = diffSARSA( problems_train, n_episodes, epsilonRL, alpha, discount_threshold, tolerance, verbose);




