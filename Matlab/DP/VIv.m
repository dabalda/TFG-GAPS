function [ PI, v, Q, n_it ] = VIv( problem, epsilon )
%VIV Value Iteration for state value functions.
%   [ PI, v, Q, n_it ] = VIv( problem, epsilon )

% Get parameters
n_states =  problem.n_states;

% Initialize v(s) arbitrarily
v = 20*rand(n_states,1)-10;

% Initialize loop variables
n_it = 0;
delta = inf;
while delta >= epsilon
    n_it = n_it+1;
    v_old = v; % Save old v
    
    % Update v with Optimal Bellman Operator
    [v, Q] = bellmanOperatorV(problem, v_old, []);
    
    % Check stability
    delta = norm(v_old-v,inf); 
end
PI = problem.getGreedyPolicy(Q, epsilon);
end

