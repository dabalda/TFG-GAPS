function [ v, Q, n_it ] = PEv( problem, PI, epsilon )
%PEV Policy Evaluation for state-value functions.
%   [ v, Q, n_it ] = PEv( problem, PI, epsilon )
%   Evaluates state value function for policy PI. Loop ends when delta <
%   epsilon.

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;
gamma =     problem.gamma;
P =         problem.Pssa;
R =         problem.Rssa;

% Initialize v(s) arbitrarily
v = 20*rand(n_states,1)-10;

% Initialize loop
delta = inf;

n_it = 0;
while delta >= epsilon
    n_it = n_it+1;
    v_old = v; % Save old v
    
    % Repeat v for each initial state and action
    v_ssa = repmat(v_old(:)',[n_states,1,n_actions]);   
    % Partial result
    v_2 = R+gamma.*v_ssa;
    % Get Q
    Q = squeeze(sum(P.*v_2, 2));
    % Update v
    v = sum(PI.*Q,2);      
    
    % Check stability
    delta = norm(v_old-v,inf);
end
end