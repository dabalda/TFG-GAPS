function [ v, Q, n_it ] = PEv( problem, PI, epsilon )
%PEV Policy Evaluation for state-value functions.
%   [ v, Q, n_it ] = PEv( problem, PI, epsilon )
%   Evaluates state value function for policy PI. Loop ends when delta <
%   epsilon.

% Get parameters
n_states =  problem.n_states;

% Initialize v(s) arbitrarily
v = 20*rand(n_states,1)-10;

% Initialize loop variables
delta = inf;
n_it = 0;
while delta >= epsilon
    n_it = n_it+1;
    v_old = v; % Save old v
    
    % Update v
    [v, Q] = bellmanOperatorV(problem, v_old, PI);  
    
    % Check stability
    delta = norm(v_old-v,inf);
end
end