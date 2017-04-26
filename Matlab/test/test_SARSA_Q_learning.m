clearvars, clc
addpath(genpath('../'))

paramRW = struct('gamma',.9,'prob_desired_right',.8,'prob_desired_left',.7,'all_states_initial',true);
problemRW = RandomWalk(paramRW);

paramRW2 = struct('gamma',.9,'prob_desired_right',.7,'prob_desired_left',.8,'all_states_initial',false);
problemRW2 = RandomWalk(paramRW2);
problemsRW = [problemRW, problemRW2];

paramCl = struct('gamma',.9,'slope_east',.0,'slope_north',.0);
problemCl = Cliff(paramCl);

verbose = true;

%%
n_episodes = 100000;
epsilonRL = .05;
alpha = .01;
discount_threshold = 0;
tolerance = 1e-3;

[ PIq, Qq ] = Q_learning( problemRW, n_episodes, epsilonRL, alpha, discount_threshold, tolerance, verbose );
%%
n_episodes = 100000;
epsilonRL = .05;
alpha = .01;
discount_threshold = 0;
tolerance = 1e-3;

[ PIs, Qs ] =      SARSA( problemRW, n_episodes, epsilonRL, alpha, discount_threshold, tolerance, verbose );
%%
epsilonPE = 1e-6;
[ Q_PEq, n ] = PEq(problemRW, PIs, epsilonPE)
[ v_PEv, n ] = PEv(problemRW, PIs, epsilonPE)

%%
Qq
Q_PEq
norm(Q_PEq - Qq, 1)
%%
Qs
Q_PEq
norm(Q_PEq - Qs, 1)

%%
epsilonPE = 1e-6;
[PI_VIv, v_PIv, Q_PIv, n] = PIv(problemRW, epsilonPE, [])
[v_PEv, Q_PEv, n] = PEv(problemRW, PI_VIv, epsilonPE)
v_PIv - v_PEv
Q_PIv - Q_PEv

%%
epsilonPE = 1e-6;
[PI_VIv, v_PIv, Q_PIv, n] = PIv(problemCl, epsilonPE, [])
[v_PEv, Q_PEv, n] = PEv(problemCl, PI_VIv, epsilonPE)
v_PIv - v_PEv
Q_PIv - Q_PEv

Q_test = getQfromV(problemCl, v_PIv);
norm(Q_test - Q_PIv)

%%
n_episodes = 100000;
epsilonRL = .05;
alpha = .01;
discount_threshold = 0;
tolerance = 1e-3;

[ PIq, Qq ] = Q_learning( problemCl, n_episodes, epsilonRL, alpha, discount_threshold, tolerance, verbose );

epsilonPE = 1e-6;
[~, Q_PEv, ~] = PEv(problemCl, PIq, epsilonPE);
norm(Q_PEv-Qq, inf)
%%
n_episodes = 100000;
epsilonRL = .05;
alpha = .01;
discount_threshold = 0;
tolerance = 1e-3;

[ PIs, Qs ] =      SARSA( problemCl, n_episodes, epsilonRL, alpha, discount_threshold, tolerance, verbose );

epsilonPE = 1e-6;
[~, Q_PEv, ~] = PEv(problemCl, PIs, epsilonPE);
norm(Q_PEv-Qs, inf)