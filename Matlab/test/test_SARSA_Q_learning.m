clearvars, clc
addpath(genpath('../'))

paramRW = struct('gamma',.9,'prob_desired_right',.8,'prob_desired_left',.7);
problemRW = RandomWalk(paramRW);

paramCl = struct('gamma',.9,'slope_east',.8,'slope_north',.7);
problemCl = Cliff(paramCl);

n_episodes = 40000;
epsilon = .1;
alpha = .05;
discount_threshold = 0;
tolerance = 1e-3;

[ PIq, Qq ] = Q_learning( problemRW, n_episodes, epsilon, alpha, discount_threshold, tolerance );
[ PIs, Qs ] =      SARSA( problemRW, n_episodes, epsilon, alpha, discount_threshold, tolerance );
Qq, Qs
PIq, PIs