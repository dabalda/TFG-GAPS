function [ PI, Q, v, n_it ] = VIq( problem, epsilon )
%VIQ Value Iteration for state-action value functions.
%   [ PI, Q, v, n_it ] = VIq( problem, epsilon )

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;

% Initialize Q(s,a) arbitrarily
Q = 20*rand(n_states,n_actions)-10;

% Initialize loop variables
n_it = 0;
delta = inf;
while delta >= epsilon
    n_it = n_it+1;
    Q_old = Q; % Save old Q
    
    % Update Q with Optimal Bellman Operator
    [Q, v] = bellmanOperatorQ(problem, Q_old);
    
    % Check stability
    delta = max(max(abs(Q_old-Q)));
end
PI = problem.getGreedyPolicy(Q, epsilon);
v = getVfromQ(PI,Q);
end