function [ Q, v, n_it ] = PEq( problem, PI, epsilon )
%PEV Policy Evaluation for state-value functions.
%   [ Q, v, n_it ] = PEv( problem, PI, epsilon )
%   Evaluates state-action value function for policy PI. Loop ends when
%   delta < epsilon.

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;

% Initialize Q(s,a) arbitrarily
Q = 20*rand(n_states,n_actions)-10;

% Initialize loop
delta = inf;

n_it = 0;
while delta >= epsilon
    n_it = n_it+1;
    Q_old = Q; % Save old Q
    
    % Update Q
    [Q, v] = bellmanOperatorQ(problem, Q_old, PI);
    
    % Check stability
    delta = max(max(abs(Q_old-Q)));
end
end